import torch
import torch.nn as nn

from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny

'''
FPN: 构建FPN特征金字塔进行加强特征提取
在特征利用部分，YoloV5提取多特征层进行目标检测，一共提取三个特征层。
三个特征层位于主干部分CSPdarknet的不同位置，分别位于中间层，中下层，底层，当输入为(640,640,3)的时候，
三个特征层的shape分别为feat1=(80,80,256)、feat2=(40,40,512)、feat3=(20,20,1024)。

在获得三个有效特征层后，我们利用这三个有效特征层进行FPN层的构建(自定义最简单的)，

从
   feat3->cbs->o3
   feat3->bottleneck->up->concat->cbs->o2
                                ->bottleneck->up->concat->cbs->o1
                                
并且引入CA注意力机制
'''
# 引入注意力机制
from nets.attentions.attention import *
attentions = [SEAttention, ECAAttention, CBAMAttention, CAAttention]

# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet',
                 pretrained=False, input_shape=[640, 640], attention_idx=0):
        super(YoloBody, self).__init__()
        depth_dict = {'s': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }  # 深度因子
        width_dict = {'s': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }  # 宽度因子
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64: base_channels
        base_depth = max(round(dep_mul * 3), 1)  # 3: Bottleneck个数  round()  四舍五入函数
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#
        self.backbone_name = backbone
        if backbone == "cspdarknet":
            # ---------------------------------------------------#
            #   生成CSPdarknet53的主干模型
            #   获得三个有效特征层(feat1, feat2, feat3)，他们的shape分别是：
            #   80,80,256
            #   40,40,512
            #   20,20,1024
            # ---------------------------------------------------#
            self.backbone = CSPDarknet(base_channels, base_depth, phi, pretrained)
        else:
            # ---------------------------------------------------#
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            # ---------------------------------------------------#
            self.backbone = {
                'convnext_tiny': ConvNeXt_Tiny,
                'convnext_small': ConvNeXt_Small,
                'swin_transfomer_tiny': Swin_transformer_Tiny,
            }[backbone](pretrained=pretrained, input_shape=input_shape)
            in_channels = {
                'convnext_tiny': [192, 384, 768],
                'convnext_small': [192, 384, 768],
                'swin_transfomer_tiny': [192, 384, 768],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)
        # end else

        # 初始化一些模块
        # 注意力机制
        self.attention_idx = attention_idx
        if 1 <= attention_idx <= 4:
            self.attention_for_up_sample1 = attentions[attention_idx - 1](512)
            self.attention_for_feat2 = attentions[attention_idx - 1](512)
            self.attention_for_up_sample2 = attentions[attention_idx - 1](256)
            self.attention_for_feat1 = attentions[attention_idx - 1](256)

        # c3
        self.c3_1 = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)  # 1024 -> 512
        self.c3_2 = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)  # 512 -> 256

        # up_sample
        self.up_sample1 = nn.Upsample(scale_factor=2, mode="nearest")  # 20*20 -> 40*40
        self.up_sample2 = nn.Upsample(scale_factor=2, mode="nearest")  # 40*40 -> 80*80

        # conv bn SiLU
        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)  # conv for feat2(40*40*512)
        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)  # conv for feat3(80*80*256)

        self.conv_for_feat2_to_p4 = Conv(base_channels * 4, base_channels * 8, 1, 1)  # 40*40*256 -> 40*40*512
        self.conv_for_feat3_to_p5 = Conv(base_channels * 8, base_channels * 16, 1, 1)  # 20*20*512 -> 20*20*1024

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone(x)
        if self.backbone_name != "cspdarknet":
            feat1 = self.conv_1x1_feat1(feat1)
            feat2 = self.conv_1x1_feat2(feat2)
            feat3 = self.conv_1x1_feat3(feat3)

        # 20, 20, 1024 -> 20, 20, 512
        P5 = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_up_sample = self.up_sample1(P5)

        if 1 <= self.attention_idx <= 4:
            P5_up_sample = self.attention_for_up_sample1(P5_up_sample)
            feat2 = self.attention_for_feat2(feat2)

        # 40, 40, 512 -> 40, 40, 1024
        P4 = torch.cat([P5_up_sample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4 = self.c3_1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4 = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_up_sample = self.up_sample2(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512

        if 1 <= self.attention_idx <= 4:
            P4_up_sample = self.attention_for_up_sample2(P4_up_sample)
            feat1 = self.attention_for_feat1(feat1)

        P3 = torch.cat([P4_up_sample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3 = self.conv_for_feat2(P3)

        # ---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        # ---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        # ---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        # ---------------------------------------------------#
        out1 = self.yolo_head_P4(self.conv_for_feat2_to_p4(P4))
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        # ---------------------------------------------------#
        out0 = self.yolo_head_P5(self.conv_for_feat3_to_p5(P5))
        return out0, out1, out2
