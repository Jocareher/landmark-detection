"""Train-only permutation parallel analysis and identity-block bootstrap for shape PCA.

Run with --help. Matrices are samples x coordinates; covariance PCA, no z-scoring.
Existing reconstruction priors are preserved; significant bases are stored separately.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.utils.synthetic_labels import parse_synthetic_landmark_label
from scripts.engine.pca_shape_prior import chi_square_critical_value


def chi_square_quantile(probability, p):
    values = np.asarray(probability, dtype=float)
    result = np.array([chi_square_critical_value(int(p), float(q)*100) for q in values.flat])
    return float(result[0]) if values.ndim == 0 else result.reshape(values.shape)


def chi_square_density(x, p):
    x = np.maximum(np.asarray(x, dtype=float), np.finfo(float).tiny)
    return np.exp((p/2-1)*np.log(x) - x/2 - (p/2)*np.log(2) - math.lgamma(p/2))


def spectrum(x):
    centered = x - x.mean(0)
    values, vectors = np.linalg.eigh(centered.T @ centered / (len(x) - 1))
    return np.maximum(values[::-1], 0), vectors[:, ::-1].T


def leading_count(values, threshold):
    """Keep a leading prefix, stopping at the first failed comparison."""
    passed = np.asarray(values) > np.asarray(threshold)
    return int(np.flatnonzero(~passed)[0]) if (~passed).any() else len(passed)


def parallel_analysis(x, identities, permutations=500, bootstraps=200, seed=20261102):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('Expected at least three finite rows of coordinates.')
    if permutations < 20 or bootstraps < 0 or len(identities) != len(x):
        raise ValueError('Need >=20 permutations, nonnegative bootstraps and one identity per row.')
    values, components = spectrum(x)
    rng = np.random.default_rng(seed)
    null = np.empty((permutations, x.shape[1]))
    for i in range(permutations):
        shuffled = np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
        null[i] = spectrum(shuffled)[0]
    # Bootstrap whole identities, keeping all their poses together. Center each resample.
    groups = [np.flatnonzero(np.asarray(identities) == identity) for identity in np.unique(identities)]
    boot = np.empty((bootstraps, x.shape[1]))
    for i in range(bootstraps):
        indices = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        boot[i] = spectrum(x[indices])[0]
    return dict(values=values, components=components, mean=x.mean(0), null=null, bootstrap=boot)


def align(shapes, reference):
    """Batched equivalent of the runtime's no-reflection similarity alignment."""
    centered = shapes - shapes.mean(1, keepdims=True)
    scale = np.linalg.norm(centered, axis=(1, 2))
    target = reference - reference.mean(0)
    target_scale = np.linalg.norm(target)
    if np.any(scale <= 1e-6):
        raise ValueError('Degenerate training shape.')
    u, _, vh = np.linalg.svd(np.einsum('nli,lj->nij', centered / scale[:, None, None], target / target_scale))
    correction = np.tile(np.eye(2), (len(shapes), 1, 1))
    correction[:, -1, -1] = np.where(np.linalg.det(u @ vh) < 0, -1, 1)
    return (target_scale / scale)[:, None, None] * (centered @ (u @ correction @ vh)) + reference.mean(0)


def attach_significant_basis(prior, result, alpha, permutations, seed):
    threshold = np.quantile(result['null'], 1 - alpha, axis=0)
    p = leading_count(result['values'], threshold)
    if p < 1:
        raise ValueError('No significant dimension: do not build a chi-square prior with zero degrees of freedom.')
    out = copy.deepcopy(prior)
    out['mahalanobis_components'] = torch.tensor(result['components'][:p].copy(), dtype=torch.float32)
    out['mahalanobis_explained_variance'] = torch.tensor(result['values'][:p], dtype=torch.float32)
    out['mahalanobis_mean_shape'] = torch.tensor(result['mean'], dtype=torch.float32).reshape(-1)
    out['parallel_analysis'] = dict(num_significant_components=p, alpha=alpha,
                                   permutations=permutations, seed=seed,
                                   method='independent_coordinate_permutation_empirical_quantile',
                                   source_split='train', covariance='unstandardized',
                                   selection='leading_prefix_until_first_failure')
    return out


def savefig(fig, directory, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def shape_mode_figure(result,p,directory,title):
    count=min(6,p);columns=min(3,count);rows=(count+columns-1)//columns
    mean=result['mean'].reshape(-1,2)
    fig,axes=plt.subplots(rows,columns,figsize=(4*columns,4*rows),squeeze=False)
    for j,a in enumerate(axes.flat):
        if j>=count:
            a.axis('off');continue
        delta=2*np.sqrt(result['values'][j])*result['components'][j].reshape(-1,2)
        for shape,color,label in [(mean-delta,'#256ba6','-2 SD'),(mean,'black','Mean'),(mean+delta,'#bd3b33','+2 SD')]:
            a.scatter(shape[:,0],shape[:,1],s=7,color=color,label=label)
        a.set_aspect('equal');a.invert_yaxis();a.set_title(f'Significant PC{j+1}');a.legend(fontsize=7)
    fig.suptitle(title+' | Coordinate variation in selected modes')
    savefig(fig,directory,'08_significant_shape_modes')


def pose_structure_figure(result,x,yaw,directory,title):
    scores=(x-result['mean'])@result['components'].T
    between=np.zeros(scores.shape[1]);total=((scores-scores.mean(0))**2).sum(0)
    for angle in np.unique(yaw):
        group=scores[yaw==angle];between+=len(group)*(group.mean(0)-scores.mean(0))**2
    eta=between/np.maximum(total,1e-30)
    pd.DataFrame(dict(component=np.arange(1,len(eta)+1),variance_fraction_associated_with_yaw=eta)).to_csv(directory/'pose_variance_decomposition.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    points=ax[0].scatter(scores[:,0],scores[:,1],c=yaw,cmap='coolwarm',s=3,alpha=.5)
    fig.colorbar(points,ax=ax[0],label='Exact yaw (degrees)')
    ax[0].set(xlabel='PC1 score',ylabel='PC2 score',title='Observed global scores colored by pose')
    ax[1].bar(np.arange(1,13),eta[:12]*100)
    ax[1].set(xlabel='Principal component',ylabel='Score variance associated with yaw (%)',ylim=(0,101),title='Between-yaw / total score variance')
    fig.suptitle(title+' | Descriptive pose association, not a causal attribution')
    savefig(fig,directory,'10_pose_structure')


def figures(result, x, prior, directory, title, alphas):
    directory.mkdir(parents=True, exist_ok=True)
    ev, null, boot = result['values'], result['null'], result['bootstrap']
    ranks = np.arange(1, len(ev) + 1)
    ps = [leading_count(ev, np.quantile(null, 1-a, axis=0)) for a in alphas]
    colors = ['#bd3b33', '#256ba6']
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a in ax:
        a.plot(ranks, null[:min(40, len(null))].T, color='#54a98b', alpha=.10)
        a.plot(ranks, ev, 'o-', color='black', ms=3, label='Observed spectrum')
        for alpha, p, color in zip(alphas, ps, colors):
            a.plot(ranks, np.quantile(null, 1-alpha, axis=0), color=color,
                   label=f'Permutation {100*(1-alpha):g}th percentile; p={p}')
        a.set(xlabel='Ordered principal component', ylabel='Eigenvalue (coordinate variance)')
    ax[0].set_title('Full spectrum'); ax[1].set(title='Leading spectrum (log scale)', yscale='log', xlim=(1, max(15, max(ps)+8)))
    limit=max(15,max(ps)+8)
    ax[1].set_ylim(max(ev[:limit].min()*.5,1e-15), max(ev[0],null[:,0].max())*1.5)
    ax[1].legend(fontsize=7); fig.suptitle(title)
    savefig(fig, directory, '01_parallel_spectrum')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for alpha, p, color in zip(alphas, ps, colors):
        ax[0].plot(ranks, ev / np.maximum(np.quantile(null, 1-alpha, axis=0), 1e-30), color=color, label=f'alpha={alpha:g}, p={p}')
        ax[1].axvline(p, color=color, ls='--', label=f'Significant p={p}')
    ax[0].axhline(1, color='black', ls='--'); ax[0].set(xlabel='Component', ylabel='Observed / null threshold', ylim=(0, 4), xlim=(1, max(25,max(ps)+8)))
    cumulative=np.cumsum(ev)/ev.sum()
    ax[1].plot(ranks, cumulative*100, color='black')
    for target in (.90,.95,.99):
        k=int(np.searchsorted(cumulative,target))+1
        ax[1].axhline(target*100,color='gray',alpha=.3); ax[1].annotate(f'{target:.0%}: K={k}',(k,target*100),fontsize=8)
    ax[1].set(xlabel='Number of components', ylabel='Cumulative explained variance (%)', ylim=(0,101))
    for a in ax:a.legend(fontsize=8)
    fig.suptitle(title); savefig(fig,directory,'02_significance_vs_variance')
    if len(boot):
        fig, ax = plt.subplots(1,2,figsize=(12,4))
        lo,hi=np.quantile(boot,[.025,.975],axis=0)
        ax[0].fill_between(ranks,lo,hi,color='#e1a17a',alpha=.5,label='Identity bootstrap 95% interval')
        ax[0].plot(ranks,ev,color='black',label='Observed'); ax[0].plot(ranks,np.quantile(null,.95,axis=0),color='#256ba6',label='Permutation 95th percentile')
        limit=max(15,max(ps)+8)
        ax[0].set(xlim=(1,limit),ylim=(max(lo[:limit].min()*.5,1e-15),max(hi[0],ev[0])*1.5),yscale='log',xlabel='Component',ylabel='Eigenvalue');ax[0].legend(fontsize=7)
        for alpha,color in zip(alphas,colors):
            threshold=np.quantile(null,1-alpha,axis=0)
            counts=[leading_count(row,threshold) for row in boot]
            ax[1].hist(counts,bins=np.arange(min(counts)-.5,max(counts)+1.5),alpha=.5,color=color,label=f'alpha={alpha:g}')
        ax[1].set(xlabel='Selected p with fixed full-data null reference',ylabel='Bootstrap repetitions'); ax[1].legend()
        fig.suptitle(title+' | Conditional stability, not an independent confidence interval for p')
        savefig(fig,directory,'03_identity_bootstrap')
    # Coordinate permutation is shown in ORIGINAL coordinate axes; then project both
    # datasets onto the same observed PC axes. Never permute the PC scores themselves.
    rng=np.random.default_rng(123)
    shuffled=np.column_stack([rng.permutation(x[:,j]) for j in range(x.shape[1])])
    pc=result['components'][:2]
    observed_scores=(x-result['mean'])@pc.T; shuffled_scores=(shuffled-result['mean'])@pc.T
    sample=rng.choice(len(x),min(1500,len(x)),replace=False)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for a,scores,label in zip(ax,[observed_scores,shuffled_scores],['Observed shapes','Independently permuted coordinates']):
        a.scatter(scores[sample,0],scores[sample,1],s=3,alpha=.3)
        a.set(title=label,xlabel='Observed PC1 score',ylabel='Observed PC2 score');a.set_aspect('equal',adjustable='datalim')
    fig.suptitle(title+' | Same projection axes for both clouds'); savefig(fig,directory,'04_shuffle_clouds')
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    distances={}
    for alpha,p,color in zip(alphas,ps,colors):
        if p==0:continue
        d=((x-result['mean'])@result['components'][:p].T)**2 / result['values'][:p]
        d=d.sum(1);distances[f'd2_alpha_{alpha:g}']=d
        grid=np.linspace(0,max(np.quantile(d,.99),chi_square_quantile(.99,p)),300)
        ax[0].hist(d,bins=50,density=True,histtype='step',color=color,label=f'Observed D², alpha={alpha:g}, p={p}')
        ax[0].plot(grid,chi_square_density(grid,p),color=color,ls='--',label=f'Chi-square({p})')
        probabilities=np.linspace(.001,.999,min(300,len(d)))
        theoretical=chi_square_quantile(probabilities,p);actual=np.quantile(d,probabilities)
        ax[1].plot(theoretical,actual,color=color,label=f'alpha={alpha:g}, p={p}')
        for coverage,style in [(.90,':'),(.95,'--')]:
            c=chi_square_quantile(coverage,p); ax[0].axvline(c,color=color,ls=style,alpha=.5)
        coverages=np.linspace(.5,.99,100)
        ax[2].plot(coverages*100,[(d<=chi_square_quantile(q,p)).mean()*100 for q in coverages],color=color,label=f'alpha={alpha:g}, p={p}')
    ax[0].set(xlabel='Squared Mahalanobis distance',ylabel='Density',title='Training distances and theoretical limits')
    upper=ax[1].get_xlim()[1];ax[1].plot([0,upper],[0,upper],'k--');ax[1].set(xlabel='Theoretical chi-square quantile',ylabel='Observed training D² quantile',title='Q-Q plot: departures from Gaussian model')
    ax[2].plot([50,99],[50,99],'k--');ax[2].set(xlabel='Nominal coverage (%)',ylabel='Observed training coverage (%)',title='Descriptive coverage, not held-out validation')
    for a in ax:a.legend(fontsize=7)
    fig.suptitle(title);savefig(fig,directory,'05_mahalanobis_calibration')
    # Whiteboard ellipsoid: contour is the 2D SECTION (other scores fixed at zero).
    p=ps[0]
    if p>=2:
        scores=(x-result['mean'])@result['components'][:p].T
        theta=np.linspace(0,2*np.pi,250)
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(scores[sample,0],scores[sample,1],c=(distances[f'd2_alpha_{alphas[0]:g}'][sample]>chi_square_quantile(.95,p)),cmap='coolwarm',s=6,alpha=.5)
        for coverage,style in [(.90,':'),(.95,'--')]:
            radii=np.sqrt(chi_square_quantile(coverage,p)*ev[:2]);ax.plot(radii[0]*np.cos(theta),radii[1]*np.sin(theta),style,label=f'{coverage:.0%} p-dimensional boundary section')
        ax.set(xlabel='PC1 score',ylabel='PC2 score',title=f'{title}\np={p}; red points exceed full-p 95% boundary');ax.legend(fontsize=7)
        savefig(fig,directory,'06_mahalanobis_ellipsoid_section')
    np.savez_compressed(directory/'distances.npz',**distances)
    recon_mean=prior['mean_shape'].numpy()
    recon_basis=prior['components'].numpy()
    reconstructed=recon_mean+(x-recon_mean)@recon_basis.T@recon_basis
    residual=((x-reconstructed)**2).mean(1)
    scores_table=pd.DataFrame(distances);scores_table['reconstruction_mse']=residual
    scores_table.to_csv(directory/'sample_geometry.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for alpha,color in zip(alphas,colors):
        ax.scatter(residual,distances[f'd2_alpha_{alpha:g}'],s=3,alpha=.25,color=color,label=f'PA alpha={alpha:g}')
    ax.set(xlabel='Reconstruction MSE in aligned space (original K)',ylabel='Significant-subspace D²',title=title+' | Two distinct geometric constraints');ax.legend()
    savefig(fig,directory,'07_reconstruction_vs_distance')
    shape_mode_figure(result,ps[0],directory,title)
    first_d=distances[f'd2_alpha_{alphas[0]:g}']
    indices=[int(np.argmax(first_d)),int(np.argmax(residual)),int(np.argsort(first_d)[len(x)//2])]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for a,idx,label in zip(axes,indices,['Largest D²','Largest reconstruction residual','Median D²']):
        shape=x[idx].reshape(-1,2);recon=reconstructed[idx].reshape(-1,2)
        a.scatter(shape[:,0],shape[:,1],s=10,label='Training shape');a.scatter(recon[:,0],recon[:,1],s=7,label='K-component reconstruction',alpha=.6)
        a.set_aspect('equal');a.invert_yaxis();a.set_title(f'{label}: row {idx}\nD²={first_d[idx]:.2f}, R={residual[idx]:.2g}');a.legend(fontsize=7)
    fig.suptitle(title+' | Training examples, not adaptation outcomes');savefig(fig,directory,'09_extreme_training_shapes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    for name in ('global','pose5','pose13'):parser.add_argument(f'--{name}-prior',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--permutations',type=int,default=500)
    parser.add_argument('--bootstraps',type=int,default=200)
    parser.add_argument('--seed',type=int,default=20261102)
    parser.add_argument('--alphas',type=float,nargs='+',default=[.05,.10])
    args=parser.parse_args()
    if len(args.alphas)!=2 or any(not 0<a<1 for a in args.alphas):parser.error('Supply two alpha levels in (0,1).')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    label_files=sorted((args.dataset_root/'train/labels').glob('*.txt'))
    if not label_files:raise FileNotFoundError('No train labels found; covariance eigenvalues alone cannot recover permutation analysis.')
    shapes=[];yaws=[];identities=[];digest=hashlib.sha256()
    for path in label_files:
        label=parse_synthetic_landmark_label(path,72)
        if label.yaw_angle is None:raise ValueError(f'Missing yaw: {path}')
        if '_yaw_' not in path.stem:raise ValueError('Identity-block bootstrap requires identity_*_yaw_* filenames.')
        shapes.append(label.landmarks*256);yaws.append(label.yaw_angle);identities.append(path.stem.split('_yaw_')[0])
        digest.update(path.name.encode());digest.update(path.read_bytes())
    shapes=np.asarray(shapes,dtype=np.float64);yaws=np.asarray(yaws);identities=np.asarray(identities)
    if not np.isfinite(shapes).all():raise ValueError('Nonfinite source coordinates.')
    rows=[];manifest=dict(dataset_root=str(args.dataset_root.resolve()),split='train',num_shapes=len(shapes),num_identities=len(np.unique(identities)),labels_sha256=digest.hexdigest(),permutations=args.permutations,bootstrap_repetitions=args.bootstraps,bootstrap_unit='identity (all poses together)',seed=args.seed,alphas=args.alphas,priors={})
    for bank in ('global','pose5','pose13'):
        source=getattr(args,f'{bank}_prior');payload=torch.load(source,map_location='cpu',weights_only=False)
        manifest['priors'][bank]=dict(path=str(source.resolve()),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if bank!='global' and payload.get('prior_mode')!=bank:raise ValueError(f'{source} is not a {bank} prior.')
        selected=np.ones(len(shapes),dtype=bool) if bank!='pose5' else np.isin(yaws,[-75,-45,0,45,75])
        aligned=align(shapes[selected],payload['global_prior']['reference_shape'].numpy().astype(float)).reshape(selected.sum(),-1)
        selected_yaw=yaws[selected];selected_ids=identities[selected]
        variants={alpha:copy.deepcopy(payload) for alpha in args.alphas}
        candidates={'global':payload['global_prior'],**payload.get('pose_priors',{})}
        for index,(label,prior) in enumerate(candidates.items()):
            mask=np.ones(len(aligned),dtype=bool) if label=='global' else selected_yaw==float(label)
            x=aligned[mask];ids=selected_ids[mask]
            if len(x)!=prior['num_samples']:raise ValueError(f'Train sample count mismatch for {bank}/{label}: {len(x)} vs {prior["num_samples"]}')
            directory=out/bank/('global' if label=='global' else f'yaw_{label}');directory.mkdir(parents=True,exist_ok=True)
            names=np.asarray([f.name for f in label_files])[selected][mask]
            pd.DataFrame(dict(row=np.arange(len(x)),label_file=names,identity=ids,yaw=selected_yaw[mask])).to_csv(directory/'sample_index.csv',index=False)
            print(f'Analyzing {bank}/{label}: {len(x)} shapes, {len(np.unique(ids))} identities',flush=True)
            cache=directory/'resampling.npz'
            if cache.exists():
                loaded=np.load(cache,allow_pickle=False)
                cache_key=json.loads(str(loaded['cache_key']))
                expected=[manifest['labels_sha256'],manifest['priors'][bank]['sha256'],args.permutations,args.bootstraps,args.seed+index]
                if cache_key!=expected:raise ValueError(f'Cache provenance mismatch: {cache}; choose a new output directory.')
                result={k:loaded[k] for k in ('values','components','mean','null','bootstrap')}
            else:
                result=parallel_analysis(x,ids,args.permutations,args.bootstraps,args.seed+index)
                cache_key=[manifest['labels_sha256'],manifest['priors'][bank]['sha256'],args.permutations,args.bootstraps,args.seed+index]
                np.savez_compressed(cache,**result,cache_key=json.dumps(cache_key))
            table=pd.DataFrame(dict(component=np.arange(1,x.shape[1]+1),observed_eigenvalue=result['values'],null_mean=result['null'].mean(0),null_q90=np.quantile(result['null'],.90,axis=0),null_q95=np.quantile(result['null'],.95,axis=0),cumulative_variance=np.cumsum(result['values'])/result['values'].sum()))
            table.to_csv(directory/'spectrum.csv',index=False)
            for alpha in args.alphas:
                threshold=np.quantile(result['null'],1-alpha,axis=0);p=leading_count(result['values'],threshold)
                if p<1:raise ValueError(f'No significant dimensions for {bank}/{label}; calibration aborted, see spectrum.csv')
                updated=attach_significant_basis(prior,result,alpha,args.permutations,args.seed+index)
                if label=='global':variants[alpha]['global_prior']=updated
                else:variants[alpha]['pose_priors'][label]=updated
                d=(((x-result['mean'])@result['components'][:p].T)**2/result['values'][:p]).sum(1)
                bs=[leading_count(v,threshold) for v in result['bootstrap']]
                row=dict(bank=bank,prior=label,n_samples=len(x),n_identities=len(np.unique(ids)),pa_alpha=alpha,p=p,reconstruction_K=prior['components'].shape[0],variance_retained_percent=result['values'][:p].sum()/result['values'].sum()*100,chi2_c90=chi_square_quantile(.90,p),chi2_c95=chi_square_quantile(.95,p),training_coverage90=np.mean(d<=chi_square_quantile(.90,p)),training_coverage95=np.mean(d<=chi_square_quantile(.95,p)),bootstrap_p_low=np.quantile(bs,.025) if bs else None,bootstrap_p_high=np.quantile(bs,.975) if bs else None,min_selected_variance=result['values'][p-1],null_half1_p=leading_count(result['values'],np.quantile(result['null'][:args.permutations//2],1-alpha,axis=0)),null_half2_p=leading_count(result['values'],np.quantile(result['null'][args.permutations//2:],1-alpha,axis=0)))
                rows.append(row);print(f'  PA alpha={alpha}: p={p}, c95={row["chi2_c95"]:.4f}',flush=True)
            figures(result,x,prior,directory,f'{bank} / {label}',args.alphas)
            if label=='global':pose_structure_figure(result,x,selected_yaw[mask],directory,f'{bank} / {label}')
        for alpha,variant in variants.items():
            variant.update(parallel_analysis_source=manifest,source_split='train')
            torch.save(variant,out/f'{bank}_parallel_alpha{round(100*alpha):02d}.pt')
    summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    overview(summary,out)
    report(summary,out)
    print(f'Finished: {out}',flush=True)


def overview(summary,out):
    fig,ax=plt.subplots(2,1,figsize=(13,8))
    keys=summary[['bank','prior']].drop_duplicates().apply(lambda r:f'{r.bank}/{r.prior}',axis=1).tolist()
    for alpha,group in summary.groupby('pa_alpha'):
        ax[0].plot(np.arange(len(group)),group.p,'o-',label=f'PA alpha={alpha:g}')
        ax[1].plot(np.arange(len(group)),group.variance_retained_percent,'o-',label=f'PA alpha={alpha:g}')
    ax[0].axhline(32,color='gray',ls='--',label='Previous K=32')
    ax[0].set(ylabel='Significant dimension p',title='Parallel analysis across global and pose-conditioned priors')
    ax[1].set(ylabel='Variance in selected subspace (%)')
    for a in ax:a.set_xticks(range(len(keys)),keys,rotation=65,ha='right');a.legend(fontsize=8)
    savefig(fig,out,'overview_dimensions')
    pivot=summary.pivot(index=['bank','prior'],columns='pa_alpha',values='training_coverage95')
    fig,ax=plt.subplots(figsize=(10,5));pivot.plot.bar(ax=ax);ax.axhline(.95,color='black',ls='--');ax.set(ylabel='Observed training coverage at theoretical 95% limit',ylim=(0,1.02),title='Does the Gaussian chi-square reference fit the training shapes?')
    savefig(fig,out,'overview_coverage')


def report(summary,out):
    g=summary[(summary.bank=='global')].iloc[0]
    conditioned=summary[summary.prior!='global']
    text=['# Significant shape dimensions and Mahalanobis calibration','',
          'Only synthetic training labels were used. Figures and all thresholds are in English. The reconstruction prior is preserved; a separate significant subspace determines both the coefficient distance and chi-square degrees of freedom.','',
          '## How to read the figures','',
          '- **01 Parallel spectrum:** black is the observed eigenvalue; green curves are independent-coordinate permutations. A leading component is retained while its eigenvalue exceeds the selected empirical null percentile. The first failed comparison stops the prefix.',
          '- **02 Significance versus variance:** an observed/null ratio above one passes the criterion. Cumulative explained variance answers a different question; 99% variance does not imply statistical significance.',
          '- **03 Identity bootstrap:** entire identities, including all their poses, are resampled together. Bands show variation of eigenvalues; the p histogram uses a fixed full-data null threshold. It is a conditional stability diagnostic, not a fully nested bootstrap confidence interval.',
          '- **04 Shuffle clouds:** coordinates are permuted before projection. Both clouds use the same observed PC axes. The null breaks cross-coordinate dependence but retains each coordinate distribution.',
          '- **05 Calibration:** compare empirical D² with chi-square(p). A Q-Q curve on the diagonal supports the Gaussian reference; departures and coverage away from the nominal percentage show model mismatch. These are training diagnostics, not independent generalization estimates.',
          '- **07–09 Geometry diagnostics:** residual-versus-distance separates reconstruction from coefficient constraints; shape modes show the effect of each significant direction; extreme examples show the largest D² and reconstruction residual. Join sample_geometry.csv to sample_index.csv by row order to identify each training label.',
          '- **06 Ellipsoid:** this is a 2D section of a p-dimensional boundary. Point colors use full-p distances, so a red point may appear inside the 2D ellipse because it is extreme in another component.','',
          '## Results and interpretation','',
          f'**Global (PA alpha={g.pa_alpha:g}):** p={int(g.p)}, retaining {g.variance_retained_percent:.2f}% coordinate variance. Inspect the shape-mode figures to identify what these directions describe; this is not a count of anatomical degrees of freedom.',
          f'**Conditioned:** exact-yaw priors retain p={conditioned.p.min()} to {conditioned.p.max()}, retaining {conditioned.variance_retained_percent.min():.2f}% to {conditioned.variance_retained_percent.max():.2f}% variance. Near-cutoff conclusions can depend on the alignment frame and permutation seed; compare the same yaw across banks and inspect bootstrap stability.',
          f'**Calibration:** global training coverage at the nominal 95% boundary is {g.training_coverage95:.2%}; conditioned coverage ranges from {conditioned.training_coverage95.min():.2%} to {conditioned.training_coverage95.max():.2%}. Inspect the histogram and Q-Q plots for departures from the Gaussian reference. Held-out validation is still needed.',
          '**Loss implication:** a smaller c is not automatically a stronger constraint, because the distance now sums fewer terms too. Reconstruction still checks deviations outside the original K-component subspace. A low-dimensional coefficient penalty leaves other within-subspace directions unconstrained by that term. No TTA accuracy improvement is established by these plots.', '']
    association=out/'global/global/pose_variance_decomposition.csv'
    if association.exists():
        frame=pd.read_csv(association)
        fractions=frame.variance_fraction_associated_with_yaw.to_numpy()
        text += [f'**Pose association:** between-yaw differences account for {fractions[0]:.2%} of global PC1 score variance and {fractions[1]:.2%} of PC2 score variance. This explains why the global selected directions chiefly describe pose-related structure. It is a descriptive association, not a causal result. See 10_pose_structure.png.', '']
    for _,r in summary.iterrows():
        text.append(f'- **{r.bank}/{r.prior}, PA alpha={r.pa_alpha:g}: p={r.p}** versus reconstruction K={r.reconstruction_K}; retains {r.variance_retained_percent:.2f}% variance. Chi-square c90={r.chi2_c90:.4f}, c95={r.chi2_c95:.4f}. Observed training coverage at c95: {r.training_coverage95:.1%}. Conditional bootstrap p interval: {r.bootstrap_p_low:g}–{r.bootstrap_p_high:g}; null half-sample selections: {r.null_half1_p}/{r.null_half2_p}.')
    text += ['', '## Decisions and limitations','',
             'PA alpha=0.05 selects dimensions using the 95th percentile of the permuted spectra; alpha=0.10 uses the 90th. This alpha is independent of the Mahalanobis coverage percentile (90 or 95). No best alpha is selected using real-image accuracy.',
             'A significant dimension is relative to this covariance-permutation null. It is not necessarily an anatomical degree of freedom: Procrustes constraints, pose and shared identities affect dependence. Permutations retain marginal coordinate distributions but destroy these dependencies. Bootstrap respects identity blocks. The analysis uses covariance PCA, without coordinate z-scoring, matching the source prior.',
             'Chi-square is an approximate Gaussian reference with parameters estimated from training data. Parallel analysis alone does not establish Gaussianity. Source coverage does not guarantee coverage on real images. Different p and covariance volumes can affect minimum-loss pose selection; this procedure does not fix that selection bias by itself.',
             'The exported .pt files keep reconstruction components and add mahalanobis_components, mahalanobis_explained_variance, mahalanobis_mean_shape and parallel_analysis metadata. Runtime sums over these p components and calculates the chi-square threshold from the same p. Manual c is rejected for these priors; set a percentile explicitly. Historical priors without PA metadata retain their previous behavior.',
             'The variance epsilon in the loss should remain below every selected variance. Inspect min_selected_variance in summary.csv; flooring a selected variance changes the theoretical chi-square relationship.',
             'No new TTA outcome is claimed here. Assess held-out identities or an external dataset and rerun the same adaptation protocol before concluding that this improves landmark accuracy.']
    (out/'README.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
