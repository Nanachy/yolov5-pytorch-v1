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

在获得三个有效特征层后，我们利用这三个有效特征层进行FPN层的构建，构建方式为：

    1. feat3=(20,20,1024)的特征层进行1次1X1卷积调整通道后获得P5，P5进行上采样UmSampling2d后与
    feat2=(40,40,512)特征层进行结合，然后使用CSPLayer进行特征提取获得P5_upsample，此时获得的特征层为(40,40,512)。
    
    2. P5_upsample=(40,40,512)的特征层进行1次1X1卷积调整通道后获得P4，P4进行上采样UmSampling2d后与
    feat1=(80,80,256)特征层进行结合，然后使用CSPLayer进行特征提取P3_out，此时获得的特征层为(80,80,256)。
    
    3. P3_out=(80,80,256)的特征层进行一次3x3卷积进行下采样，下采样后与P4堆叠，然后使用CSPLayer进行特征提取P4_out，
    此时获得的特征层为(40,40,512)。
    
    4. P4_out=(40,40,512)的特征层进行一次3x3卷积进行下采样，下采样后与P5堆叠，然后使用CSPLayer进行特征提取P5_out，
    此时获得的特征层为(20,20,1024)。
    
特征金字塔可以将不同shape的特征层进行特征融合，有利于提取出更好的特征。
'''


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
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
            #   获得三个有效特征层，他们的shape分别是：
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

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        # down sample
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

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
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4 = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4 = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3 = self.conv3_for_upsample2(P3)

        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        # ---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        # ---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        # ---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        # ---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        # ---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2
