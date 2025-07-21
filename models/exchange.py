import torch
from torch import nn

from models.mynet_parts import log_feature_2


class TimeAttention2(nn.Module):

    def __init__(self, channels):
        super(TimeAttention2, self).__init__()
        self.ch = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        attn_channels = channels // 4
        attn_channels = max(attn_channels, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, attn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_channels),
            nn.ReLU(),
            nn.Conv2d(attn_channels, channels * 2, kernel_size=1, bias=False),
        )


    def forward(self, x, log=False, module_name = None, img_name = None):
        y = torch.split(x, split_size_or_sections=self.ch, dim=1)
        x1 = y[0]
        x2 = y[1]

        if log:
            log_list = [x1, x2]
            feature_name_list = ['x1_before', 'x2_before']
            log_feature_2(log_list=log_list, module_name=module_name,
                        feature_name_list=feature_name_list,
                        img_name=img_name, module_output=True)

        x = self.avg_pool(x)
        y = self.mlp(x)
        B, C, H, W = y.size()
        x1_attn, x2_attn = y.reshape(B, 2, C // 2, H, W).transpose(0, 1)
        x1_attn = torch.sigmoid(x1_attn)
        x2_attn = torch.sigmoid(x2_attn)
        x1 = x1 * x1_attn + x1
        x2 = x2 * x2_attn + x2

        if log:
            log_list = [x1, x2]
            feature_name_list = ['x1_after', 'x2_after']
            log_feature_2(log_list=log_list, module_name=module_name,
                        feature_name_list=feature_name_list,
                        img_name=img_name, module_output=True)

        return x1, x2

