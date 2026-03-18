from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
TransferMode = Literal["feature_extractor", "fine_tuning"]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Create a 3x3 convolution with padding and no bias."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """Standard residual block used in HRNet branches."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """Initialize one residual basic block."""
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to one feature tensor."""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    """Bottleneck residual block used in the HRNet stem stage."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """Initialize one bottleneck residual block."""
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the bottleneck block to one feature tensor."""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class HighResolutionModule(nn.Module):
    """Multi-branch HRNet module with feature fusion across resolutions."""

    def __init__(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> None:
        """Construct one HRNet module with its branches and fuse layers."""
        super().__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = list(num_inchannels)
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches, block, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _check_branches(
        num_branches: int,
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
    ) -> None:
        """Validate that all branch configuration lists have matching lengths."""
        if (
            num_branches != len(num_blocks)
            or num_branches != len(num_inchannels)
            or num_branches != len(num_channels)
        ):
            raise ValueError("HRNet branch configuration lengths do not match.")

    def _make_one_branch(
        self,
        branch_index: int,
        block: type[nn.Module],
        num_blocks: list[int],
        num_channels: list[int],
        stride: int = 1,
    ) -> nn.Sequential:
        """Build one residual branch inside the HRNet module."""
        downsample = None
        expected_channels = num_channels[branch_index] * block.expansion
        if stride != 1 or self.num_inchannels[branch_index] != expected_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    expected_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(expected_channels, momentum=BN_MOMENTUM),
            )

        layers: list[nn.Module] = [
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        ]
        self.num_inchannels[branch_index] = expected_channels
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: list[int],
        num_channels: list[int],
    ) -> nn.ModuleList:
        """Build all residual branches for the current HRNet module."""
        return nn.ModuleList(
            [
                self._make_one_branch(i, block, num_blocks, num_channels)
                for i in range(num_branches)
            ]
        )

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        """Create the layers that fuse information across different resolutions."""
        if self.num_branches == 1:
            return None

        fuse_layers: list[nn.ModuleList] = []
        for target_branch in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer: list[nn.Module | None] = []
            for source_branch in range(self.num_branches):
                if source_branch > target_branch:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                self.num_inchannels[source_branch],
                                self.num_inchannels[target_branch],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            BatchNorm2d(
                                self.num_inchannels[target_branch], momentum=BN_MOMENTUM
                            ),
                        )
                    )
                elif source_branch == target_branch:
                    fuse_layer.append(None)
                else:
                    conv3x3s: list[nn.Module] = []
                    for k in range(target_branch - source_branch):
                        out_channels = (
                            self.num_inchannels[target_branch]
                            if k == target_branch - source_branch - 1
                            else self.num_inchannels[source_branch]
                        )
                        blocks = [
                            nn.Conv2d(
                                self.num_inchannels[source_branch],
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                        ]
                        if k != target_branch - source_branch - 1:
                            blocks.append(nn.ReLU(inplace=True))
                        conv3x3s.append(nn.Sequential(*blocks))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> list[int]:
        """Return the output channel count for each branch."""
        return self.num_inchannels

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply branch processing and cross-resolution feature fusion."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for branch_index in range(self.num_branches):
            x[branch_index] = self.branches[branch_index](x[branch_index])

        fused_outputs: list[torch.Tensor] = []
        assert self.fuse_layers is not None
        for target_branch in range(len(self.fuse_layers)):
            y = x[0] if target_branch == 0 else self.fuse_layers[target_branch][0](x[0])
            for source_branch in range(1, self.num_branches):
                if source_branch == target_branch:
                    y = y + x[source_branch]
                elif source_branch > target_branch:
                    y = y + F.interpolate(
                        self.fuse_layers[target_branch][source_branch](
                            x[source_branch]
                        ),
                        size=(x[target_branch].shape[2], x[target_branch].shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    y = y + self.fuse_layers[target_branch][source_branch](
                        x[source_branch]
                    )
            fused_outputs.append(self.relu(y))
        return fused_outputs


BLOCKS_DICT = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HRNetW18Backbone(nn.Module):
    """HRNetV2-W18 backbone used as the shared feature extractor."""

    STEM_INPLANES = 64
    STAGE2 = {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 2,
        "NUM_BLOCKS": [4, 4],
        "NUM_CHANNELS": [18, 36],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    }
    STAGE3 = {
        "NUM_MODULES": 4,
        "NUM_BRANCHES": 3,
        "NUM_BLOCKS": [4, 4, 4],
        "NUM_CHANNELS": [18, 36, 72],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    }
    STAGE4 = {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 4,
        "NUM_BLOCKS": [4, 4, 4, 4],
        "NUM_CHANNELS": [18, 36, 72, 144],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    }

    def __init__(self) -> None:
        """Construct the HRNet stem and all multi-resolution stages."""
        super().__init__()
        self.inplanes = self.STEM_INPLANES
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, self.inplanes, 64, 4)

        stage2_block = BLOCKS_DICT[self.STAGE2["BLOCK"]]
        stage2_num_channels = [
            channels * stage2_block.expansion
            for channels in self.STAGE2["NUM_CHANNELS"]
        ]
        self.transition1 = self._make_transition_layer([256], stage2_num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.STAGE2, stage2_num_channels
        )

        stage3_block = BLOCKS_DICT[self.STAGE3["BLOCK"]]
        stage3_num_channels = [
            channels * stage3_block.expansion
            for channels in self.STAGE3["NUM_CHANNELS"]
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, stage3_num_channels
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.STAGE3, stage3_num_channels
        )

        stage4_block = BLOCKS_DICT[self.STAGE4["BLOCK"]]
        stage4_num_channels = [
            channels * stage4_block.expansion
            for channels in self.STAGE4["NUM_CHANNELS"]
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, stage4_num_channels
        )
        self.stage4, pre_stage_channels = self._make_stage(
            self.STAGE4, stage4_num_channels, multi_scale_output=True
        )
        self.final_inp_channels = sum(pre_stage_channels)

    def _make_transition_layer(
        self, num_channels_pre_layer: list[int], num_channels_cur_layer: list[int]
    ) -> nn.ModuleList:
        """Create transition layers between successive HRNet stages."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers: list[nn.Module | None] = []
        for current_branch in range(num_branches_cur):
            if current_branch < num_branches_pre:
                if (
                    num_channels_cur_layer[current_branch]
                    != num_channels_pre_layer[current_branch]
                ):
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[current_branch],
                                num_channels_cur_layer[current_branch],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            BatchNorm2d(
                                num_channels_cur_layer[current_branch],
                                momentum=BN_MOMENTUM,
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s: list[nn.Module] = []
                for step_index in range(current_branch + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = (
                        num_channels_cur_layer[current_branch]
                        if step_index == current_branch - num_branches_pre
                        else in_channels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(
        self,
        block: type[nn.Module],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build a residual stage composed of repeated blocks."""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers: list[nn.Module] = [block(inplanes, planes, stride, downsample)]
        current_inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(current_inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(
        self,
        layer_config: dict[str, object],
        num_inchannels: list[int],
        multi_scale_output: bool = True,
    ) -> tuple[nn.Sequential, list[int]]:
        """Build a full HRNet stage composed of one or more modules."""
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = BLOCKS_DICT[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules: list[nn.Module] = []
        current_num_inchannels = list(num_inchannels)
        for module_index in range(num_modules):
            reset_multi_scale_output = (
                multi_scale_output or module_index < num_modules - 1
            )
            hr_module = HighResolutionModule(
                num_branches=num_branches,
                block=block,
                num_blocks=num_blocks,
                num_inchannels=current_num_inchannels,
                num_channels=num_channels,
                fuse_method=fuse_method,
                multi_scale_output=reset_multi_scale_output,
            )
            modules.append(hr_module)
            current_num_inchannels = hr_module.get_num_inchannels()
        return nn.Sequential(*modules), current_num_inchannels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fused high-resolution feature map produced by the backbone."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list: list[torch.Tensor] = []
        for branch_index in range(self.STAGE2["NUM_BRANCHES"]):
            x_list.append(
                self.transition1[branch_index](x)
                if self.transition1[branch_index] is not None
                else x
            )
        y_list = self.stage2(x_list)

        x_list = []
        for branch_index in range(self.STAGE3["NUM_BRANCHES"]):
            x_list.append(
                self.transition2[branch_index](y_list[-1])
                if self.transition2[branch_index] is not None
                else y_list[branch_index]
            )
        y_list = self.stage3(x_list)

        x_list = []
        for branch_index in range(self.STAGE4["NUM_BRANCHES"]):
            x_list.append(
                self.transition3[branch_index](y_list[-1])
                if self.transition3[branch_index] is not None
                else y_list[branch_index]
            )
        stage4_outputs = self.stage4(x_list)

        height, width = stage4_outputs[0].shape[2], stage4_outputs[0].shape[3]
        upsampled_features = [stage4_outputs[0]]
        for feature_map in stage4_outputs[1:]:
            upsampled_features.append(
                F.interpolate(
                    feature_map,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat(upsampled_features, dim=1)


class HRNetLandmarkVisibility(nn.Module):
    """HRNet-based multitask model with heatmap and visibility heads."""

    FINAL_CONV_KERNEL = 1

    def __init__(self, num_landmarks: int = 72) -> None:
        """Initialize the multitask model and its task-specific heads."""
        super().__init__()
        self.num_landmarks = num_landmarks
        self.backbone = HRNetW18Backbone()
        in_channels = self.backbone.final_inp_channels
        final_padding = 1 if self.FINAL_CONV_KERNEL == 3 else 0

        self.landmark_head = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False
            ),
            BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                num_landmarks,
                kernel_size=self.FINAL_CONV_KERNEL,
                stride=1,
                padding=final_padding,
                bias=True,
            ),
        )
        self.visibility_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BatchNorm2d(in_channels // 2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels // 2,
                num_landmarks,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.AdaptiveAvgPool2d(1),
        )
        self._initialize_new_heads()

    def _initialize_new_heads(self) -> None:
        """Initialize the newly created task heads with lightweight random weights."""
        for module in list(self.landmark_head.modules()) + list(
            self.visibility_head.modules()
        ):
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared HRNet features before the task-specific heads."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return landmark heatmaps and visibility logits for an input batch."""
        features = self.forward_features(x)
        landmark_heatmaps = self.landmark_head(features)
        visibility_logits = self.visibility_head(features).flatten(start_dim=1)
        return {"heatmaps": landmark_heatmaps, "visibility_logits": visibility_logits}

    def load_official_hrnet_pretrained(
        self, pretrained_path: str, verbose: bool = True
    ) -> dict[str, list[str]]:
        """Load matching backbone weights from an official HRNet checkpoint."""
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_path}"
            )

        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        checkpoint_state = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )

        current_state = self.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        skipped_keys: list[str] = []
        for key, value in checkpoint_state.items():
            normalized_key = key[7:] if key.startswith("module.") else key
            if normalized_key.startswith("head"):
                skipped_keys.append(key)
                continue
            target_key = f"backbone.{normalized_key}"
            if (
                target_key in current_state
                and current_state[target_key].shape == value.shape
            ):
                filtered_state[target_key] = value
            else:
                skipped_keys.append(key)

        missing_before_load = [
            key for key in current_state.keys() if key not in filtered_state
        ]
        self.load_state_dict(filtered_state, strict=False)
        if verbose:
            print(f"Loaded {len(filtered_state)} tensors from checkpoint.")
            print(
                f"Skipped {len(skipped_keys)} checkpoint tensors that do not match the backbone."
            )
            print(f"Backbone keys missing from checkpoint: {len(missing_before_load)}")
        return {
            "loaded_keys": sorted(filtered_state.keys()),
            "skipped_keys": sorted(skipped_keys),
            "missing_keys": sorted(missing_before_load),
        }

    def set_transfer_learning_mode(
        self,
        mode: TransferMode,
        num_unfrozen_stages: int = 0,
        unfreeze_stem: bool = False,
    ) -> None:
        """Freeze or unfreeze backbone stages according to the selected transfer setup."""
        if mode not in {"feature_extractor", "fine_tuning"}:
            raise ValueError(f"Invalid transfer mode: {mode}")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.landmark_head.parameters():
            parameter.requires_grad = True
        for parameter in self.visibility_head.parameters():
            parameter.requires_grad = True

        if mode == "feature_extractor":
            return

        num_unfrozen_stages = max(0, min(num_unfrozen_stages, 4))
        modules_to_unfreeze: list[nn.Module] = []
        if num_unfrozen_stages >= 1:
            modules_to_unfreeze.extend(
                [self.backbone.transition3, self.backbone.stage4]
            )
        if num_unfrozen_stages >= 2:
            modules_to_unfreeze.extend(
                [self.backbone.transition2, self.backbone.stage3]
            )
        if num_unfrozen_stages >= 3:
            modules_to_unfreeze.extend(
                [self.backbone.transition1, self.backbone.stage2]
            )
        if num_unfrozen_stages >= 4:
            modules_to_unfreeze.append(self.backbone.layer1)
        if unfreeze_stem:
            modules_to_unfreeze.extend(
                [
                    self.backbone.conv1,
                    self.backbone.bn1,
                    self.backbone.conv2,
                    self.backbone.bn2,
                ]
            )

        for module in modules_to_unfreeze:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def get_trainable_parameter_names(self) -> list[str]:
        """Return the names of all parameters currently marked as trainable."""
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
