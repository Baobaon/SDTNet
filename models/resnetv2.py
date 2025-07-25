# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn

from mmseg.models.backbones import ResNet

class TimeAttention2(nn.Module):

    def __init__(self, channels):
        super(TimeAttention2, self).__init__()
        self.ch = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        attn_channels = channels // 16
        attn_channels = max(attn_channels, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, attn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_channels),
            nn.ReLU(),
            nn.Conv2d(attn_channels, channels * 2, kernel_size=1, bias=False),
        )

    def forward(self, x1, x2):

        y = torch.split(torch.cat([x1, x2], dim=1), split_size_or_sections=self.ch, dim=1)
        x1 = y[0]
        x2 = y[1]
        x = self.avg_pool(torch.cat([x1, x2], dim=1))
        y = self.mlp(x)
        B, C, H, W = y.size()
        x1_attn, x2_attn = y.reshape(B, 2, C // 2, H, W).transpose(0, 1)
        x1_attn = torch.sigmoid(x1_attn)
        x2_attn = torch.sigmoid(x2_attn)
        x1 = x1 * x1_attn + x1
        x2 = x2 * x2_attn + x2
        return x1, x2


class TwoIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity1 = nn.Identity()
        self.identity2 = nn.Identity()

    def forward(self, x1, x2):
        x1 = self.identity1(x1)
        x2 = self.identity2(x2)
        return x1, x2

class IA_ResNet(ResNet):
    """Interaction ResNet backbone.

    Args:
        interaction_cfg (Sequence[dict]): Interaction strategies for the stages.
            The length should be the same as `num_stages`. The details can be
            found in `opencd/models/utils/interaction_layer.py`.
            Default: (None, None, None, None).
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): Resnet stages, normally 4. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: (1, 2, 2, 2).
        dilations (Sequence[int]): Dilation of each stage.
            Default: (1, 1, 1, 1).
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            When conv_cfg is None, cfg will be set to dict(type='Conv2d').
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (dict | None): Dictionary to construct and config DCN conv layer.
            When dcn is not None, conv_cfg must be None. Default: None.
        stage_with_dcn (Sequence[bool]): Whether to set DCN conv for each
            stage. The length of stage_with_dcn is equal to num_stages.
            Default: (False, False, False, False).
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.

            - position (str, required): Position inside block to insert plugin,
            options: 'after_conv1', 'after_conv2', 'after_conv3'.

            - stages (tuple[bool], optional): Stages to apply plugin, length
            should be same as 'num_stages'.
            Default: None.
        multi_grid (Sequence[int]|None): Multi grid dilation rates of last
            stage. Default: None.
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        # >>> from opencd.models import IA_ResNet
        # >>> import torch
        # >>> self = IA_ResNet(depth=18)
        # >>> self.eval()
        # >>> inputs = torch.rand(1, 3, 32, 32)
        # >>> level_outputs = self.forward(inputs, inputs)
        # >>> for level_out in level_outputs:
        # ...     print(tuple(level_out.shape))
        # (1, 128, 8, 8)
        # (1, 256, 4, 4)
        # (1, 512, 2, 2)
        # (1, 1024, 1, 1)
    """

    def __init__(self,
                 interaction_cfg=(None, 64, 128, None, None),
                 **kwargs):
        super().__init__(**kwargs)

        # cross-correlation
        self.ccs = []
        for ia_cfg in interaction_cfg:
            if ia_cfg == None:
                self.ccs.append(TwoIdentity())
            else:
                self.ccs.append(TimeAttention2(ia_cfg))
        self.ccs = nn.ModuleList(self.ccs)

    def forward(self, x1, x2):
        """Forward function."""

        def _stem_forward(x):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            # x = self.maxpool(x)
            return x

        x1 = _stem_forward(x1)
        x2 = _stem_forward(x2)
        t1_outs = []
        t2_outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            x1, x2 = self.ccs[i](x1, x2)
            if i in self.out_indices:
                t1_outs.append(x1)
                t2_outs.append(x2)
        return tuple(t1_outs), tuple(t2_outs)



class IA_ResNetV1c(IA_ResNet):
    """ResNetV1c variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs. For more details please refer to `Bag
    of Tricks for Image Classification with Convolutional Neural Networks
    <https://arxiv.org/abs/1812.01187>`_.
    """

    def __init__(self, **kwargs):
        super(IA_ResNetV1c, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)



class IA_ResNetV1d(IA_ResNet):
    """ResNetV1d variant described in [1]_.
    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(IA_ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)

