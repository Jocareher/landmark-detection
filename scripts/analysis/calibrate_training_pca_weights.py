"""Scale Mahalanobis to reconstruction while preserving existing loss weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def recommend_relative_losses(table, ratio=1.0):
    """Match mean weighted loss VALUES, including batches where the hinge is zero."""
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('The Mahalanobis/reconstruction target ratio must be positive.')
    key=('lambda_pca_reconstruction' if 'lambda_pca_reconstruction' in table
         else 'lambda_pca_projection')
    if table[key].nunique()!=1:
        raise ValueError('Use one diagnostic run with a fixed reconstruction weight.')
    wr=float(table[key].iloc[0])
    if not np.isfinite(wr) or wr <= 0:
        raise ValueError('Keep your EXISTING positive reconstruction weight for this diagnostic; do not set both weights to zero.')
    r=table.pca_reconstruction_loss.to_numpy()
    h=table.pca_mahalanobis_penalty.to_numpy()
    weights=table.images.to_numpy() if 'images' in table else np.ones(len(table))
    valid=np.isfinite(r)&np.isfinite(h)&np.isfinite(weights)&(weights>0)&(r>=0)&(h>=0)
    if not valid.any():raise ValueError('No finite nonnegative raw loss measurements.')
    mr=float(np.average(r[valid],weights=weights[valid]))
    mh=float(np.average(h[valid],weights=weights[valid]))
    return dict(weight_mode='loss_relative_to_reconstruction',probes=int(valid.sum()),
        active_mahalanobis_probes=int((h[valid]>0).sum()),
        lambda_pca_reconstruction=wr,lambda_pca_mahalanobis=(ratio*wr*mr/mh if mh>0 else None),
        mean_raw_reconstruction=mr,mean_raw_mahalanobis=mh,
        target_mahalanobis_to_reconstruction=ratio,
        caveat='Preserves reconstruction and all other loss weights. Scales average loss VALUES, not gradients. Zero-hinge batches are included. If every hinge is zero its weight cannot be estimated. Fix the proposed weight for a pilot; do not divide by the current batch loss during training. Mean matching does not guarantee equal gradients or prevent spikes.')


def recommend_independent(table, reconstruction_fraction=.05, mahalanobis_fraction=.05):
    """Estimate each explicit weight against the SAME non-PCA gradient reference."""
    if not (0 < reconstruction_fraction <= 1 and 0 < mahalanobis_fraction <= 1):
        raise ValueError('Target gradient fractions must lie in (0,1].')
    if len(table[['lambda_pca_reconstruction','lambda_pca_mahalanobis']].drop_duplicates())!=1:
        raise ValueError('Use one fixed-weight diagnostic run; do not mix experiments.')
    go=table.other_losses_grad_norm.to_numpy()
    result=dict(probes=len(table),weight_mode='independent',
                target_reconstruction_fraction=reconstruction_fraction,
                target_mahalanobis_fraction=mahalanobis_fraction)
    for name,column,target in [
            ('lambda_pca_reconstruction','raw_reconstruction_grad_norm',reconstruction_fraction),
            ('lambda_pca_mahalanobis','raw_mahalanobis_grad_norm',mahalanobis_fraction)]:
        g=table[column].to_numpy()
        use=np.isfinite(g)&np.isfinite(go)&(g>0)&(go>0)
        result[name]=float(np.median(target*go[use]/g[use])) if use.any() else None
        result[name+'_usable_probes']=int(use.sum())
    result['caveat']='Starting weights, not validated optima. Each term targets 5% of the non-PCA gradient by default. Zero-hinge batches cannot calibrate Mahalanobis. Use matched 3–5 epoch pilots and held-out validation NME. Probe batches are correlated. No automatic weight updates.'
    return result


def recommend(table, mahalanobis_fraction=.1, pca_fraction=.1):
    if not (0 < mahalanobis_fraction <= 1 and 0 < pca_fraction <= 1):
        raise ValueError('Target gradient fractions must lie in (0,1].')
    if table.beta.nunique()!=1:
        raise ValueError('Use probes from a single fixed beta; do not mix weight experiments.')
    gr=table.raw_reconstruction_grad_norm.to_numpy()
    gh=table.raw_mahalanobis_grad_norm.to_numpy()
    gp=table.raw_pca_grad_norm.to_numpy()
    go=table.other_losses_grad_norm.to_numpy()
    active=np.isfinite(gr)&np.isfinite(gh)&(gh>0)&(gr>0)
    usable=np.isfinite(gp)&np.isfinite(go)&(gp>0)&(go>0)
    return dict(
        probes=len(table),active_mahalanobis_probes=int(active.sum()),
        beta_used=float(table.beta.iloc[0]),
        target_mahalanobis_to_reconstruction=mahalanobis_fraction,
        beta_candidate=(float(np.median(mahalanobis_fraction*gr[active]/gh[active])) if active.any() else None),
        target_pca_to_other=pca_fraction,
        lambda_pca_candidate_at_current_beta=(float(np.median(pca_fraction*go[usable]/gp[usable])) if usable.any() else None),
        caveat='Heuristic starting weights, not an optimum. Choose beta first, then remeasure the combined PCA gradient at that beta before choosing lambda. Zero-hinge probes cannot calibrate beta. Gradients use all currently trainable parameters, before AMP scaling/clipping. Probe batches are correlated. Validate candidates with held-out validation NME, not BabyLand test performance.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--method',choices=['loss-relative','gradient-independent','gradient-legacy'],default='loss-relative')
    p.add_argument('--mahalanobis-to-reconstruction',type=float,default=1.0,
                   help='Target mean weighted Mahalanobis / weighted reconstruction; 1 matches their mean magnitudes.')
    p.add_argument('--mahalanobis-fraction',type=float,default=.05)
    p.add_argument('--reconstruction-fraction',type=float,default=.05)
    p.add_argument('--pca-fraction',type=float,default=.1)
    a=p.parse_args();files=sorted(a.log_dir.glob('train_epoch_*.csv'))
    if not files:raise FileNotFoundError('No gradient probe CSV files found')
    table=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    if a.method=='loss-relative':
        result=recommend_relative_losses(table,a.mahalanobis_to_reconstruction)
    elif a.method=='gradient-independent':
        result=recommend_independent(table,a.reconstruction_fraction,a.mahalanobis_fraction)
    else:
        result=recommend(table,a.mahalanobis_fraction,a.pca_fraction)
    result['source_files']=[str(f.resolve()) for f in files]
    path=a.log_dir/'weight_starting_points.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False))
    if result.get('weight_mode') in {'independent','loss_relative_to_reconstruction'}:
        print('\nStarting YAML values (null means insufficient measurements):')
        for key in ['lambda_pca_reconstruction','lambda_pca_mahalanobis']:
            print(f'{key}: {json.dumps(result[key])}')


if __name__=='__main__':main()
