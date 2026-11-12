"""Signal selection, permutation invariants, and consistent runtime dimensions."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from scripts.analysis.pca_parallel_analysis import parallel_analysis, leading_count, attach_significant_basis, align
from scripts.engine.pca_shape_prior import (
    align_shape_to_reference_torch, compute_pca_projection_terms,
    pca_mahalanobis_threshold, chi_square_critical_value,
)
from test_pca_mahalanobis import example


def test_parallel_analysis_recovers_two_signals_and_preserves_trace():
    rng=np.random.default_rng(4)
    scores=rng.normal(size=(350,2))
    loadings=np.zeros((2,12));loadings[0,:6]=1;loadings[1,6:]=1
    x=scores@loadings+rng.normal(scale=.05,size=(350,12))
    ids=np.repeat(np.arange(175),2)
    result=parallel_analysis(x,ids,permutations=50,bootstraps=10,seed=12)
    assert leading_count(result['values'],np.quantile(result['null'],.95,axis=0))==2
    np.testing.assert_allclose(result['null'].sum(1),result['values'].sum(),rtol=1e-10)
    # Whole-row permutations would preserve the spectrum and fail this check.
    assert result['values'][0]>np.quantile(result['null'][:,0],.95)*3
    shifted=parallel_analysis(x+np.arange(12)*10,ids,50,10,12)
    for key in ('values','null','bootstrap'):
        np.testing.assert_allclose(result[key],shifted[key],rtol=1e-8,atol=1e-10)


def test_leading_prefix_edge_cases():
    assert leading_count([1,2],[2,3])==0
    assert leading_count([3,2],[2,1])==2
    assert leading_count([3,1,4],[2,2,2])==1


def test_vectorized_alignment_matches_runtime():
    rng=np.random.default_rng(3)
    shapes=rng.normal(size=(5,7,2));reference=rng.normal(size=(7,2))
    observed=align(shapes,reference)
    expected=torch.stack([align_shape_to_reference_torch(torch.tensor(s),torch.tensor(reference)) for s in shapes]).numpy()
    np.testing.assert_allclose(observed,expected,rtol=2e-5,atol=2e-6)


def test_significant_distance_and_threshold_share_p_reconstruction_unchanged():
    x,payload=example();prior=payload['global_prior']
    rng=np.random.default_rng(3)
    result=dict(values=np.linspace(.001,.0001,10),components=np.eye(10),mean=prior['mean_shape'].numpy(),null=np.tile(np.array([.0005,.0004]+[1.]*8),(20,1)))
    selected=attach_significant_basis(prior,result,.05,20,3)
    assert selected['parallel_analysis']['num_significant_components']==2
    torch.testing.assert_close(selected['components'],prior['components'])
    new=deepcopy(payload);new['global_prior']=selected
    oldterms=compute_pca_projection_terms(x,payload,'aligned')
    x.requires_grad_()
    terms=compute_pca_projection_terms(x,new,'aligned',mahalanobis_enabled=True,mahalanobis_percentile=95)
    aligned=align_shape_to_reference_torch(x[0],selected['reference_shape']).reshape(1,-1)
    coefficients=(aligned-selected['mahalanobis_mean_shape'])@selected['mahalanobis_components'].T
    torch.testing.assert_close(terms['mahalanobis_squared'],(coefficients.square()/selected['mahalanobis_explained_variance']).sum())
    expected=pca_mahalanobis_threshold(selected,10,95)
    assert expected==pytest.approx(chi_square_critical_value(2,95))
    torch.testing.assert_close(terms['reconstruction'],oldterms['reconstruction'])
    torch.testing.assert_close(terms['mahalanobis_penalty'],(terms['mahalanobis_squared']-expected).clamp_min(0))
    gradient,=torch.autograd.grad(terms['loss'],x)
    assert torch.isfinite(gradient).all()
    with pytest.raises(ValueError,match='manual c'):
        pca_mahalanobis_threshold(selected,10,None)
    selected['parallel_analysis']['num_significant_components']=3
    with pytest.raises(ValueError,match='dimensions'):
        pca_mahalanobis_threshold(selected,10,95)
