import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from model_warehouse.src.models._blocks import Conv1x1
from model_warehouse.src.models.p2v import VideoEncoder, SimpleDecoder, DecBlock
from models.SAM import SupervisedAttentionModule
from models.mynet_parts import (Conv_BN_ReLU, CGSU, Encoder_Block, DPFA, Decoder_Block,
                                Changer_channel_exchange, log_feature, CBAM, Encoder_lastBlock)
from models.exchange import TimeAttention2
from models.resnetv2 import IA_ResNetV1c



class SimpleFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleFNN, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
class MyNet(nn.Module):

    def __init__(self):
        super().__init__()

        channel_list = [32, 64, 128, 256, 512]
        # encoder
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=1),
                                       CGSU(in_channel=channel_list[0]),
                                       CGSU(in_channel=channel_list[0]),
                                       )
        self.en_block2 = Encoder_Block(in_channel=channel_list[0], out_channel=channel_list[1])
        self.en_block3 = Encoder_Block(in_channel=channel_list[1], out_channel=channel_list[2])
        self.en_block4 = Encoder_Block(in_channel=channel_list[2], out_channel=channel_list[3])
        # self.en_block5 = Encoder_Block(in_channel=channel_list[3], out_channel=channel_list[4])
        self.en_block5 = Encoder_lastBlock(in_channel=channel_list[3], out_channel=channel_list[4])

        self.exchange1 = TimeAttention2(64)
        self.exchange2 = TimeAttention2(128)

        self.channel_exchange4 = Changer_channel_exchange()

        # decoder
        self.de_block1 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.de_block2 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.de_block3 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        # dpfa
        self.dpfa1 = DPFA(in_channel=channel_list[4])
        self.dpfa2 = DPFA(in_channel=channel_list[3])
        self.dpfa3 = DPFA(in_channel=channel_list[2])
        self.dpfa4 = DPFA(in_channel=channel_list[1])

        # change path
        # the change block is the same as decoder block
        # the change block is used to fuse former and latter change features

        self.change_block4 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.change_block3 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.change_block2 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        self.seg_out1 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)
        self.seg_out2 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)

        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_change = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

        # 3D Encoder
        self.pe = nn.Conv3d(3, 3, (1, 3, 3), padding=(0, 1, 1), groups=3)
        self.mlp = SimpleFNN(2,12,6)
        self.encoder_3d = VideoEncoder(3, [64, 128])
        self.conv_out_v = Conv1x1(512, 1)
        enc_chs_v = [256, 512]
        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2 * ch, ch, norm=True, act=True)
                for ch in enc_chs_v
            ]
        )
        self.fuse1 = DecBlock(256, 512, 256)
        self.fuse2 = DecBlock(128, 256, 128)
        self.attn1 = CBAM(256)
        self.attn2 = CBAM(512)

        # init parameters
        # using pytorch default init is enough
        # self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)

    def forward(self, t1, t2, log=False, img_name=None, label=None):
        """ Model forward and log feature if :obj:`log: is True.

        If :obj:`log` is True, some module output and model output will be saved.

        To be more specific, module output will be saved in folder named
        `:obj:`module_input_feature_name`_:obj:`module_name`-
        :obj:`module_input_feature_name`_:obj:`module_name`- ...
        _:obj:`log_feature_name``. For example, module output saved folder could be named
        `t1_1_en_block2-x_cbam-spatial_weight`.

        Module output in saved folder will have the same name as corresponding input image.

        Model output saved folder will be simply named `model_:obj:`log_feature_name``. For example,
        it could be `model_seg_out_1`.

        Model output in saved folder will have the same name as corresponding input image.

        :obj:`seg_out1` and :obj:`seg_out2` could be used in loss function to train model better,
        and :obj:`change_out` is the prediction of model.

        Parameter:
            t1(tensor): input t1 image.
            t2(tensor): input t2 image.
            log(bool): if True, log output of module and model.
            img_name(tensor): name of input image.

        Return:
            change_out(tensor): change prediction of model.
            seg_out1(tensor): auxiliary change prediction through t1 decoder branch.
            seg_out2(tensor): auxiliary change prediction through t2 decoder branch.
        """
        # 3d encoder
        F_3d = self.pe(torch.cat([t1.unsqueeze(2), t2.unsqueeze(2)], dim=2))
        F_3d = self.mlp(F_3d.transpose(-1, -3)).transpose(-1, -3)
        fmaps = self.encoder_3d(F_3d)
        fmaps.pop(0)

        # fmaps[0] = self.convs_video[0](self.tem_aggr(fmaps[0]))
        # fmaps[0], mask0 = self.sam1(fmaps[0])
        # fmaps[1] = self.convs_video[1](self.tem_aggr(fmaps[1]))
        # fmaps[1], mask1 = self.sam2(fmaps[1])
        # aux_3d = []
        # aux_3d.append(F.interpolate(mask0, size=(256, 256)))
        # aux_3d.append(F.interpolate(mask1, size=(256, 256)))

        for i, f_3d in enumerate(fmaps):
            attn = getattr(self, f'attn{i+1}')
            fmaps[i] = self.convs_video[i](self.tem_aggr(f_3d))
            fmaps[i] = attn(fmaps[i])
        aux_3d = self.conv_out_v(fmaps[-1])
        aux_3d = F.interpolate(aux_3d, size=(256, 256))

        # encoder
        t1_1 = self.en_block1(t1)
        t2_1 = self.en_block1(t2)

        t1_2 = self.en_block2(t1_1)
        t2_2 = self.en_block2(t2_1)

        t1_2, t2_2 = self.exchange1(
            torch.cat([t1_2, t2_2], dim=1), log=False, img_name=img_name, module_name = '1exchange'
        )

        t1_3 = self.en_block3(t1_2)
        t2_3 = self.en_block3(t2_2)

        t1_3, t2_3 = self.exchange2(
            torch.cat([t1_3, t2_3], dim=1), log=False, img_name=img_name, module_name = '2exchange'
        )

        t1_4 = self.en_block4(t1_3)
        t2_4 = self.en_block4(t2_3)
        t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)

        t1_5 = self.en_block5(t1_4)
        t2_5 = self.en_block5(t2_4)


        de1_5 = t1_5
        de2_5 = t2_5

        de1_4 = self.de_block1(de1_5, t1_4)
        de2_4 = self.de_block1(de2_5, t2_4)

        de1_3 = self.de_block2(de1_4, t1_3)
        de2_3 = self.de_block2(de2_4, t2_3)

        de1_2 = self.de_block3(de1_3, t1_2)
        de2_2 = self.de_block3(de2_3, t2_2)

        seg_out1 = self.seg_out1(de1_2)
        seg_out2 = self.seg_out2(de2_2)

        if log:
            change_5 = self.dpfa1(de1_5, de2_5, log=log, module_name='de1_5_de2_5_dpfa1',
                                  img_name=img_name)

            change_4 = self.change_block4(change_5, self.dpfa2(de1_4, de2_4, log=log, module_name='de1_4_de2_4_dpfa2',
                                                               img_name=img_name))
            change_4 = self.fuse1(change_4, fmaps[1])

            change_3 = self.change_block3(change_4, self.dpfa3(de1_3, de2_3, log=log, module_name='de1_3_de2_3_dpfa3',
                                                               img_name=img_name))
            change_3 = self.fuse2(change_3, fmaps[0])

            change_2 = self.change_block2(change_3, self.dpfa4(de1_2, de2_2, log=log, module_name='de1_2_de2_2_dpfa4',
                                                               img_name=img_name))
        else:
            change_5 = self.dpfa1(de1_5, de2_5)

            change_4 = self.change_block4(change_5, self.dpfa2(de1_4, de2_4))
            change_4 = self.fuse1(change_4, fmaps[1])

            change_3 = self.change_block3(change_4, self.dpfa3(de1_3, de2_3))
            change_3 = self.fuse2(change_3, fmaps[0])

            change_2 = self.change_block2(change_3, self.dpfa4(de1_2, de2_2))



        change = self.upsample_x2(change_2)
        change_out = self.conv_out_change(change)


        # log_feature(log_list=[change_out, seg_out1, seg_out2], module_name='model_LEVIR_ABMF',
        #             feature_name_list=['change_out', 'seg_out1', 'seg_out2'],
        #             img_name=img_name, module_output=False, labels=label)

        return change_out, seg_out1, seg_out2, aux_3d




# x = torch.randn([1, 3, 256, 256]).cuda()
# model = DPCD().cuda()
# from thop import profile
# flops, params = profile(model, (x, x))
# print('FLOPs: ', flops)
# print('Params: ', params)
