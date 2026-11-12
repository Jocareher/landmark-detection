import numpy as np
import pytest
import torch

from scripts.analysis.build_grouped_pose5_prior import (
    GROUPS, REPRESENTATIVES, class_for_yaw, fit_grouped,
)
from scripts.engine.tta_losses import resolve_gt_pose_labels


def test_angle_groups_keep_all_thirteen_yaws_and_babyland_class_semantics():
    angles=[-75,-65,-55,-45,-30,-15,0,15,30,45,55,65,75]
    assert [class_for_yaw(y) for y in angles]==[0,0,1,1,1,2,2,2,3,3,3,4,4]
    assert class_for_yaw(0)==2  # Zero degrees is NOT BabyLand class 0.
    assert resolve_gt_pose_labels(list(range(5)),5)==['-75','-45','0','45','75']
    for y in angles:
        assert class_for_yaw(-y)==4-class_for_yaw(y)
    with pytest.raises(ValueError):class_for_yaw(20)


def test_grouped_fit_pools_intermediate_angles_instead_of_discarding_them():
    torch.manual_seed(8)
    yaw=np.repeat([a for v in GROUPS.values() for a in v],3)
    base=torch.tensor([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    shapes=base+torch.randn(len(yaw),4,2)*.06
    shapes[:,0,0]+=torch.tensor(yaw)/300
    payload,aligned,classes=fit_grouped(shapes,yaw,components=2)
    assert len(aligned)==39 and payload['num_excluded_samples']==0
    assert payload['prior_mode']=='pose5_grouped'
    for c,values in GROUPS.items():
        prior=payload['pose_priors'][str(REPRESENTATIVES[c])]
        assert prior['num_samples']==len(values)*3
        assert prior['source_yaw_angles']==list(values)
        np.testing.assert_allclose(prior['mean_shape'].numpy(),aligned[classes==c].mean(0),atol=1e-7)
