import math
import torch
from torch import nn

__all__ = ['ECAAttention', 'SEAttention', 'CBAMAttention', 'CAAttention']

'''
SE（Squeeze-and-Excitation）

优点：
可以通过学习自适应的通道权重，使得模型更加关注有用的通道信息。
缺点：
SE注意力机制只考虑了通道维度上的注意力，无法捕捉空间维度上的注意力，适用于通道数较多的场景，
但对于通道数较少的情况可能不如其他注意力机制。
'''


class SEAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        # b,c,w,h=b,c,1,1
        avg = self.avg_pool(x).view([b, c])

        # b,c -> b,c//ratio ->b,c,1,1
        fc = self.fc(avg).view([b, c, 1, 1])
        return x * fc


'''
ECA（Efficient Channel Attention）
优点：
    可以同时考虑通道维度和空间维度上的注意力，对于特征图尺寸较大的场景下，计算效率较高。
缺点：
    需要额外的计算，因此对于较小的特征图，可能会有较大的计算开销。
'''


class ECAAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECAAttention, self).__init__()
        t = int(abs(math.log(channel, 2) + b) / gamma)
        kernel_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, w, h = x.size()
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.sigmoid(self.conv(avg)).view([b, c, 1, 1])
        return out * x


'''
CBAM（Convolutional Block Attention Module）
优点：
    结合了卷积和注意力机制，可以从空间和通道两个方面上对图像进行关注。
缺点：
    需要更多的计算资源，计算复杂度更高。
'''


# 定义通道注意力机制
class channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # h w 变为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])

        return out * x


# 空间注意力机制
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        padding = 7 // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        # 返回值； values值  indices索引
        print('torch.max: ', torch.max(x, dim=1, keepdim=True))
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)

        out = torch.cat([max_pool_out, avg_pool_out], dim=1)
        out = self.sigmoid(self.conv(out))

        return out * x


# CBAM
class CBAMAttention(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAMAttention, self).__init__()
        self.channel_attention = channel_attention(channel, ratio)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


'''
CA注意力机制:
该文章的作者认为现有的注意力机制（如CBAM、SE）在求取通道注意力的时候，通道的处理一般是采用全局最大池化/平均池化，
这样会损失掉物体的空间信息。作者期望在引入通道注意力机制的同时，引入空间注意力机制，
作者提出的注意力机制将位置信息嵌入到了通道注意力中。

CA注意力的实现如图所示，可以认为分为两个并行阶段：

将输入特征图分别在为宽度和高度两个方向分别进行全局平均池化，分别获得在宽度和高度两个方向的特征图。
假设输入进来的特征层的形状为[C, H, W]，在经过宽方向的平均池化后，获得的特征层shape为[C, H, 1]，
此时我们将特征映射到了高维度上；在经过高方向的平均池化后，获得的特征层shape为[C, 1, W]，此时我们将特征映射到了宽维度上。
然后将两个并行阶段合并，将宽和高转置到同一个维度，然后进行堆叠，将宽高特征合并在一起，此时我们获得的特征层为：
[C, 1, H+W]，利用卷积+标准化+激活函数获得特征。

之后再次分开为两个并行阶段，再将宽高分开成为：[C, 1, H]和[C, 1, W]，之后进行转置。
获得两个特征层[C, H, 1]和[C, 1, W]。
然后利用1x1卷积调整通道数后取sigmoid获得宽高维度上的注意力情况。乘上原有的特征就是CA注意力机制。
'''


class CAAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAAttention, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
