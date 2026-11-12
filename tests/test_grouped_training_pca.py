from copy import deepcopy
import numpy as np
import pytest
import torch

from scripts.analysis.build_grouped_pose5_prior import fit_grouped
from scripts.engine.pca_shape_prior import (
    grouped_training_prior_labels, compute_pca_projection_terms,
    softargmax_heatmaps_to_image_coords,
)
from scripts.engine.landmark_losses import MeanSquaredHeatmapLoss
from scripts.engine.losses import compute_multitask_loss
from scripts.engine.loss_calibration import training_loss_probe
from scripts.engine.train import run_epoch
from scripts.analysis.calibrate_training_pca_weights import recommend


def fixture():
    torch.manual_seed(17)
    yaw=np.repeat([-75,-65,-55,-45,-30,-15,0,15,30,45,55,65,75],3)
    base=torch.tensor([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.],[-.3,.2]])
    x=base+torch.randn(len(yaw),5,2)*.08
    x[:,0,0]+=torch.tensor(yaw)/300
    prior,_,_=fit_grouped(x,yaw,2)
    for c,key in enumerate(prior['pose_priors']):
        p=prior['pose_priors'][key];count=1+c%2
        p.update(mahalanobis_components=p['components'][:count].clone(),
                 mahalanobis_explained_variance=torch.full((count,),1e-5),
                 mahalanobis_mean_shape=p['mean_shape'].clone(),
                 parallel_analysis={'num_significant_components':count})
    y=torch.tensor([-65.,-55.,0.,55.,65.])
    batch={'heatmaps':torch.rand(5,5,8,8),'visibility':torch.ones(5,5),'yaw_angle':y}
    return prior,batch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heatmaps=torch.nn.Parameter(torch.randn(5,5,8,8))
        self.vis=torch.nn.Parameter(torch.zeros(5,5))
    def forward(self,images):
        return {'heatmaps':self.heatmaps,'visible_heatmaps':self.heatmaps*.9,'visibility_logits':self.vis}


def kwargs(prior):
    return dict(pca_shape_prior=prior,pca_training_prior_mode='gt_grouped',pca_loss_space='aligned',
                pca_mahalanobis_enabled=True,pca_mahalanobis_percentile=95.,
                lambda_pca_projection=3.,pca_mahalanobis_weight=.002)


def test_grouped_training_uses_label_prior_and_correct_outer_weights():
    prior,batch=fixture();model=Model();outputs=model(None)
    losses=compute_multitask_loss(outputs,batch,MeanSquaredHeatmapLoss(),torch.nn.BCEWithLogitsLoss(),
                                  image_height=32,image_width=32,**kwargs(prior))
    coords=softargmax_heatmaps_to_image_coords(outputs['heatmaps'],32,32)
    labels=grouped_training_prior_labels(batch['yaw_angle'],prior,5)
    assert labels==['-75','-45','0','45','75']
    expected=[]
    for i,label in enumerate(labels):
        expected.append(compute_pca_projection_terms(coords[i:i+1],
            {'alignment':prior['alignment'],'global_prior':prior['pose_priors'][label]},
            'aligned',mahalanobis_enabled=True,mahalanobis_percentile=95.,mahalanobis_weight=.002))
    for target,source in [('pca_loss','loss'),('pca_reconstruction_loss','reconstruction'),('pca_mahalanobis_penalty','mahalanobis_penalty')]:
        torch.testing.assert_close(losses[target],torch.stack([e[source] for e in expected]).mean())
    torch.testing.assert_close(losses['weighted_pca_loss'],losses['weighted_pca_reconstruction_loss']+losses['weighted_pca_total_mahalanobis_loss'])
    # Diagnostic backprop must not populate .grad or change the actual optimization gradient.
    before=torch.autograd.grad(losses['total_loss'],model.heatmaps,retain_graph=True)[0]
    probe=training_loss_probe(losses,model,3.,.002)
    assert all(np.isfinite(v) for v in probe.values()) and model.heatmaps.grad is None
    losses['total_loss'].backward()
    torch.testing.assert_close(before,model.heatmaps.grad)
    assert probe['weighted_mahalanobis_grad_norm']==pytest.approx(.006*probe['raw_mahalanobis_grad_norm'])


def test_wrong_bank_missing_yaw_and_class_indices_fail_instead_of_silent_global():
    prior,batch=fixture()
    for yaw in [None,[float('nan')]*5,[0.,1.,2.,3.,4.]]:
        with pytest.raises(ValueError):grouped_training_prior_labels(yaw,prior,5)
    exact=deepcopy(prior);exact['pose_selection']='exact_yaw'
    with pytest.raises(ValueError,match='explicit yaw_groups'):grouped_training_prior_labels(batch['yaw_angle'],exact,5)


def test_training_and_validation_propagate_yaw_and_write_gradient_probes(tmp_path):
    prior,batch=fixture();batch['image']=torch.rand(5,3,32,32)
    model=Model();optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)
    args=dict(model=model,dataloader=[batch],device=torch.device('cpu'),
              heatmap_loss_fn=MeanSquaredHeatmapLoss(),visibility_loss_fn=torch.nn.BCEWithLogitsLoss(),
              use_amp=False,**kwargs(prior))
    log=tmp_path/'probes.csv'
    train=run_epoch(**args,optimizer=optimizer,pca_gradient_monitor_interval=1,pca_gradient_log_path=log)
    assert log.exists() and 'raw_mahalanobis_grad_norm' in log.read_text()
    val=run_epoch(**args,training=False,pca_gradient_monitor_interval=1,pca_gradient_log_path=tmp_path/'no.csv')
    assert not (tmp_path/'no.csv').exists()
    assert np.isfinite(train['weighted_pca_loss']) and np.isfinite(val['weighted_pca_loss'])


def test_calibration_does_not_invent_beta_when_hinge_is_inactive():
    import pandas as pd
    table=pd.DataFrame(dict(beta=[0.,0.],raw_reconstruction_grad_norm=[2.,2.],
        raw_mahalanobis_grad_norm=[0.,0.],raw_pca_grad_norm=[2.,2.],other_losses_grad_norm=[10.,10.]))
    result=recommend(table)
    assert result['beta_candidate'] is None
    assert result['lambda_pca_candidate_at_current_beta']==pytest.approx(.5)
    table.raw_mahalanobis_grad_norm=[20.,20.]
    assert recommend(table)['beta_candidate']==pytest.approx(.01)


@pytest.mark.parametrize('explicit_weights',[False,True])
def test_full_training_checkpoint_reports_per_group_p_and_thresholds(tmp_path,explicit_weights):
    from scripts.engine.train import train_model
    prior,batch=fixture();prior_path=tmp_path/'grouped.pt';torch.save(prior,prior_path)
    batch['image']=torch.rand(5,3,32,32)
    model=Model();opt=torch.optim.SGD(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1)
    options=kwargs(prior);options.pop('pca_shape_prior')
    if explicit_weights:
        options.update(lambda_pca_reconstruction=0.,lambda_pca_mahalanobis=0.)
    loader=torch.utils.data.DataLoader([{k:v[i] for k,v in batch.items()} for i in range(5)],batch_size=5)
    result=train_model(model,loader,loader,opt,scheduler,MeanSquaredHeatmapLoss(),
        torch.nn.BCEWithLogitsLoss(),torch.device('cpu'),1,tmp_path/'run',
        pca_prior_path=prior_path,use_wandb=False,use_amp=False,visualize_every_n_epochs=0,
        pca_gradient_monitor_interval=1,**options)
    assert result['pca_training_prior_mode']=='gt_grouped'
    assert result['pca_mahalanobis_effective_dimensions']=={'-75':1,'-45':2,'0':1,'45':2,'75':1}
    saved=torch.load(tmp_path/'run/last_model.pth',weights_only=False)
    assert saved['pca_training_prior_mode']=='gt_grouped'
    assert (tmp_path/'run/loss_calibration/train_epoch_0001.csv').exists()
    if explicit_weights:
        assert saved['lambda_pca_reconstruction']==saved['lambda_pca_mahalanobis']==0
        assert result['history']['train'][0]['pca_mahalanobis_penalty']>0
        assert result['history']['val'][0]['pca_mahalanobis_penalty']>0
        assert result['history']['train'][0]['weighted_pca_loss']==0


def test_cli_accepts_training_routing_and_probe_interval(monkeypatch):
    from scripts.main import parse_args
    monkeypatch.setattr('sys.argv',['train','--pca-training-prior-mode','gt_grouped',
                                    '--pca-gradient-monitor-interval','10',
                                    '--lambda-pca-reconstruction','2',
                                    '--lambda-pca-mahalanobis','.001'])
    args=parse_args()
    assert args.pca_training_prior_mode=='gt_grouped' and args.pca_gradient_monitor_interval==10
    assert args.lambda_pca_reconstruction==2 and args.lambda_pca_mahalanobis==.001


def test_independent_weights_override_legacy_product_and_can_measure_at_zero():
    prior,batch=fixture();model=Model();outputs=model(None)
    options=kwargs(prior)
    values=compute_multitask_loss(outputs,batch,MeanSquaredHeatmapLoss(),torch.nn.BCEWithLogitsLoss(),
        image_height=32,image_width=32,lambda_pca_reconstruction=7.,lambda_pca_mahalanobis=.013,**options)
    torch.testing.assert_close(values['weighted_pca_reconstruction_loss'],7*values['pca_reconstruction_loss'])
    torch.testing.assert_close(values['weighted_pca_total_mahalanobis_loss'],.013*values['pca_mahalanobis_penalty'])
    torch.testing.assert_close(values['weighted_pca_loss'],7*values['pca_reconstruction_loss']+.013*values['pca_mahalanobis_penalty'])
    zero=compute_multitask_loss(outputs,batch,MeanSquaredHeatmapLoss(),torch.nn.BCEWithLogitsLoss(),
        image_height=32,image_width=32,lambda_pca_reconstruction=0.,lambda_pca_mahalanobis=0.,
        compute_pca_diagnostics=True,**options)
    assert zero['weighted_pca_loss']==0 and zero['pca_mahalanobis_penalty']>0
    probe=training_loss_probe(zero,model,3.,.002,0.,0.)
    assert probe['raw_mahalanobis_grad_norm']>0 and probe['weighted_pca_grad_norm']==0
    assert probe['weighted_reconstruction_grad_norm']==probe['weighted_mahalanobis_grad_norm']==0
    from scripts.engine.losses import resolve_pca_training_weights
    with pytest.raises(ValueError,match='BOTH'):resolve_pca_training_weights(1.,1.,2.,None)


def test_independent_calibration_returns_two_lambdas_from_same_reference():
    import pandas as pd
    from scripts.analysis.calibrate_training_pca_weights import recommend_independent
    table=pd.DataFrame(dict(lambda_pca_reconstruction=[0.,0.],lambda_pca_mahalanobis=[0.,0.],
        raw_reconstruction_grad_norm=[2.,2.],raw_mahalanobis_grad_norm=[20.,0.],other_losses_grad_norm=[10.,10.]))
    result=recommend_independent(table)
    assert result['lambda_pca_reconstruction']==pytest.approx(.25)
    assert result['lambda_pca_mahalanobis']==pytest.approx(.025)
    assert result['lambda_pca_mahalanobis_usable_probes']==1
    table.raw_mahalanobis_grad_norm=[0.,0.]
    assert recommend_independent(table)['lambda_pca_mahalanobis'] is None


def test_loss_value_calibration_preserves_reconstruction_and_includes_zero_hinges():
    import pandas as pd
    from scripts.analysis.calibrate_training_pca_weights import recommend_relative_losses
    table=pd.DataFrame(dict(lambda_pca_reconstruction=[2.,2.],pca_reconstruction_loss=[.001,.003],
                            pca_mahalanobis_penalty=[0.,20.],images=[1,3]))
    result=recommend_relative_losses(table)
    assert result['lambda_pca_reconstruction']==2
    assert result['mean_raw_reconstruction']==pytest.approx(.0025)
    assert result['mean_raw_mahalanobis']==15
    assert result['lambda_pca_mahalanobis']*15==pytest.approx(2*.0025)
    table.pca_mahalanobis_penalty=[0.,0.]
    assert recommend_relative_losses(table)['lambda_pca_mahalanobis'] is None
    table.lambda_pca_reconstruction=[0.,0.]
    with pytest.raises(ValueError,match='EXISTING'):recommend_relative_losses(table)
