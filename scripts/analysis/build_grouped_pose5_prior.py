"""Review exact synthetic yaw versus BabyLand classes and refit grouped PCA/PA.

All prior fits and resampling use SynBaby train labels only. BabyLand crops are
used solely for a documented visual review, never to estimate a prior or p.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from scripts.analysis.pca_parallel_analysis import (
    parallel_analysis, attach_significant_basis, figures, leading_count,
    chi_square_quantile, pose_structure_figure,
)
from scripts.engine.pca_shape_prior import build_pose_pca_shape_prior_payload
from scripts.utils.synthetic_labels import parse_synthetic_landmark_label
from scripts.utils.orientation import ORIENTATION_CLASS_TO_NAME


# Representatives are bank keys, NOT the exact yaw of every member.
GROUPS = {0: (-75, -65), 1: (-55, -45, -30), 2: (-15, 0, 15),
          3: (30, 45, 55), 4: (65, 75)}
REPRESENTATIVES = {0: -75, 1: -45, 2: 0, 3: 45, 4: 75}


def class_for_yaw(yaw):
    matches = [c for c, values in GROUPS.items() if float(yaw) in values]
    if len(matches) != 1:
        raise ValueError(f'No unique grouped class for exact synthetic yaw {yaw!r}')
    return matches[0]


def fit_grouped(shapes, yaws, components=32):
    classes = np.array([class_for_yaw(y) for y in yaws])
    representatives = torch.tensor([REPRESENTATIVES[int(c)] for c in classes])
    payload, aligned, by_pose = build_pose_pca_shape_prior_payload(
        shapes, representatives, mode='pose5', num_components=components)
    payload.update(prior_mode='pose5_grouped', pose_selection='explicit_yaw_groups',
                   source_split='train', yaw_groups={str(c): list(v) for c,v in GROUPS.items()},
                   orientation_class_to_prior={str(c): str(v) for c,v in REPRESENTATIVES.items()})
    for c, angles in GROUPS.items():
        p=payload['pose_priors'][str(REPRESENTATIVES[c])]
        p.update(orientation_class=c, orientation_name=ORIENTATION_CLASS_TO_NAME[c],
                 source_yaw_angles=list(angles), yaw_angle_is_representative=True)
    return payload, aligned.numpy().reshape(len(aligned), -1), classes


def visual_review(syn, baby, paths, yaws, identities, out):
    angles=sorted(set(yaws));ids=np.unique(identities)
    chosen=ids[np.linspace(0,len(ids)-1,3,dtype=int)]
    manifest={'synthetic':[], 'babyland':[]}
    sheet=Image.new('RGB',(len(angles)*160,3*190),'white');draw=ImageDraw.Draw(sheet)
    for row, identity in enumerate(chosen):
        for col, angle in enumerate(angles):
            ix=np.flatnonzero((identities==identity)&(yaws==angle));assert len(ix)==1
            label=paths[int(ix[0])];image=syn/'train/images'/(label.stem+'.jpg')
            im=Image.open(image).convert('RGB');im.thumbnail((158,158))
            x,y=col*160,row*190;sheet.paste(im,(x,y));c=class_for_yaw(angle)
            draw.text((x+4,y+160),f'{angle:+g} deg / class {c}',fill='black')
            manifest['synthetic'].append(dict(identity=str(identity),yaw=float(angle),class_idx=c,image=str(image),label=str(label)))
    sheet.save(out/'synthetic_13_angles_grouped.jpg')
    rng=np.random.default_rng(20261112);counts=Counter()
    labels=[]
    for label in sorted((baby/'labels').glob('*.txt')):
        raw=label.read_text().splitlines()[0].strip()
        if raw not in {'0','1','2','3','4'}:raise ValueError(f'Invalid BabyLand class header in {label}')
        if not label.name.startswith('flip_'):
            labels.append((label,int(raw)));counts[int(raw)]+=1
    sheet=Image.new('RGB',(8*180,5*210),'white');draw=ImageDraw.Draw(sheet)
    for c in GROUPS:
        group=[p for p,cc in labels if cc==c]
        for j,k in enumerate(rng.choice(len(group),8,replace=False)):
            label=group[int(k)];image=baby/'all_detections/images'/(label.stem+'__det_000.jpg')
            im=Image.open(image).convert('RGB');im.thumbnail((178,178))
            x,y=j*180,c*210;sheet.paste(im,(x,y));draw.text((x+4,y+180),f'class {c}: {label.stem}',fill='black')
            manifest['babyland'].append(dict(class_idx=c,image=str(image),label=str(label)))
    sheet.save(out/'babyland_classes_0_to_4.jpg')
    manifest['babyland_original_counts']=dict(counts)
    (out/'visual_review_samples.json').write_text(json.dumps(manifest,indent=2))
    return dict(counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--babyland-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--permutations',type=int,default=500)
    parser.add_argument('--bootstraps',type=int,default=200)
    parser.add_argument('--seed',type=int,default=20261112)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    paths=sorted((args.dataset_root/'train/labels').glob('*.txt'))
    if not paths:raise ValueError('No train labels')
    shapes=[];yaws=[];identities=[];digest=hashlib.sha256()
    for p in paths:
        a=parse_synthetic_landmark_label(p,72)
        class_for_yaw(a.yaw_angle)
        shapes.append(a.landmarks*256);yaws.append(a.yaw_angle)
        identities.append(p.stem.split('_yaw_')[0]);digest.update(p.name.encode());digest.update(p.read_bytes())
    shapes=torch.tensor(np.asarray(shapes));yaws=np.asarray(yaws);identities=np.asarray(identities)
    assert np.isfinite(shapes.numpy()).all()
    counts=visual_review(args.dataset_root,args.babyland_root,paths,yaws,identities,out)
    payload,x,classes=fit_grouped(shapes,yaws)
    assert payload['num_excluded_samples']==0
    manifest=dict(dataset_root=str(args.dataset_root.resolve()),split='train',
        labels_sha256=digest.hexdigest(),num_shapes=len(x),num_identities=len(np.unique(identities)),
        exact_yaw_counts={str(int(a)):int((yaws==a).sum()) for a in sorted(set(yaws))},
        class_to_name=ORIENTATION_CLASS_TO_NAME,yaw_groups=GROUPS,representative_keys=REPRESENTATIVES,
        babyland_original_class_counts=counts,babyland_use='Visual class review only; excluded from all PCA and PA fits',
        grouping_rationale='Explicit symmetric nearest-representative grouping for the 13 available angles; ±55 maps to ±45',
        permutations=args.permutations,bootstrap_repetitions=args.bootstraps,
        bootstrap_unit='identity, all member angles resampled together',seed=args.seed,
        common_alignment='Shared global Procrustes reference, no reflection',
        interpretation='Visual correspondence proposal, not a statistically optimal or validated class boundary')
    payload['grouped_prior_source']=manifest
    torch.save(payload,out/'pose5_grouped_k32.pt')
    pd.DataFrame(dict(label_file=[p.name for p in paths],identity=identities,exact_yaw=yaws,
                     class_idx=classes,prior_key=[str(REPRESENTATIVES[int(c)]) for c in classes])).to_csv(out/'sample_index.csv',index=False)
    variants={a:deepcopy(payload) for a in [.05,.10]};rows=[];yaw_rows=[]
    groups=[('global',np.ones(len(x),bool)),*[(str(REPRESENTATIVES[c]),classes==c) for c in GROUPS]]
    for index,(key,mask) in enumerate(groups):
        folder=out/('global' if key=='global' else 'prior_'+key);folder.mkdir(exist_ok=True)
        prior=payload['global_prior'] if key=='global' else payload['pose_priors'][key]
        xx=x[mask].astype(np.float64);ids=identities[mask]
        print(f'Analyzing {key}: {len(xx)} samples, {len(np.unique(ids))} identities',flush=True)
        cache=folder/'resampling.npz';cache_key=json.dumps([manifest['labels_sha256'],GROUPS,args.permutations,args.bootstraps,args.seed+index],sort_keys=True)
        if cache.exists():
            saved=np.load(cache);assert str(saved['cache_key'])==cache_key
            result={k:saved[k] for k in ['values','components','mean','null','bootstrap']}
        else:
            result=parallel_analysis(xx,ids,args.permutations,args.bootstraps,args.seed+index)
            np.savez_compressed(cache,**result,cache_key=cache_key)
        np.testing.assert_allclose(result['null'].sum(1),result['values'].sum(),rtol=1e-7)
        if key!='global':
            scores=(xx-result['mean'])@result['components'].T
            total=((scores-scores.mean(0))**2).sum(0);between=np.zeros(144)
            for angle in np.unique(yaws[mask]):
                ss=scores[yaws[mask]==angle]
                between+=len(ss)*(ss.mean(0)-scores.mean(0))**2
            yaw_rows.extend(dict(prior=key,component=j+1,
                                 between_yaw_fraction=float(between[j]/max(total[j],1e-30)))
                            for j in range(6))
        pd.DataFrame(dict(component=np.arange(1,145),observed=result['values'],null_q90=np.quantile(result['null'],.9,axis=0),null_q95=np.quantile(result['null'],.95,axis=0))).to_csv(folder/'spectrum.csv',index=False)
        for alpha in variants:
            updated=attach_significant_basis(prior,result,alpha,args.permutations,args.seed+index)
            if key=='global':variants[alpha]['global_prior']=updated
            else:variants[alpha]['pose_priors'][key]=updated
            threshold=np.quantile(result['null'],1-alpha,axis=0);p=leading_count(result['values'],threshold)
            d=(((xx-result['mean'])@result['components'][:p].T)**2/result['values'][:p]).sum(1)
            bs=np.array([leading_count(v,threshold) for v in result['bootstrap']])
            rows.append(dict(prior=key,n_samples=len(xx),n_identities=len(np.unique(ids)),alpha=alpha,p=p,K=32,
                variance_retained_percent=100*result['values'][:p].sum()/result['values'].sum(),
                chi2_c90=chi_square_quantile(.90,p),chi2_c95=chi_square_quantile(.95,p),
                training_coverage90=float(np.mean(d<=chi_square_quantile(.90,p))),training_coverage95=float(np.mean(d<=chi_square_quantile(.95,p))),
                bootstrap_p_low=float(np.quantile(bs,.025)),bootstrap_p_high=float(np.quantile(bs,.975))))
            print(f'  alpha={alpha}: p={p}, c95={rows[-1]["chi2_c95"]:.4f}',flush=True)
            # Reconstruction remains exactly the 32-mode fit while PA changes only its coefficient basis.
            torch.testing.assert_close(updated['components'],prior['components'])
        figures(result,xx,prior,folder,'Grouped pose5 / '+key,[.05,.10])
        if key=='global':pose_structure_figure(result,xx,yaws[mask],folder,'Grouped pose5 / global')
    for alpha,variant in variants.items():
        variant['parallel_analysis_source']=manifest
        torch.save(variant,out/f'pose5_grouped_parallel_alpha{round(alpha*100):02d}.pt')
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(yaw_rows).to_csv(out/'within_group_yaw_association.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Finished grouped priors and parallel analysis',flush=True)


if __name__=='__main__':main()
