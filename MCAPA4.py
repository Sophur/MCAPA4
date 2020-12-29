import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#import get_dct_weights
#from get_dct_weights import get_dct_weights

# from core.blocks import M_Encoder
# from core.blocks import M_Conv

# from core.blocks import  M_Decoder_my_10
# from guided_filter_pytorch.guided_filter_attention import FastGuidedFilter_attention

class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=None):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # , dilation=3,padding=3
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),

        )

        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)

        self.relu = nn.ReLU(inplace=True)

        self.calayer = CALayer(out_ch)
        self.palayer = PALayer(out_ch)

    def forward(self, x):
        residual = x
        out = self.conv(x)

        residual = self.conv1x1(residual)

        out += residual

        out = self.relu(out)

        out = self.calayer(out)
        out = self.palayer(out)

        out = out * residual

        return out


class DoubleConv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv_up, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MCAPA4(nn.Module):
    def __init__(self, n_classes=2, bn=True, BatchNorm=False):
        super(MCAPA4, self).__init__()

        self.conv1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
        self.down1 = ConvRelu(1, 64)

        self.conv2 = DoubleConv(64 + 32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down2 = ConvRelu(1, 128)

        self.conv3 = DoubleConv(128 + 64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.down3 = ConvRelu(1, 256)

        self.conv4 = DoubleConv(256 + 128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(256, 512)
        # 逆卷积
        self.up6 = nn.ConvTranspose2d(1184, 512, 2, stride=2)
        self.conv6 = DoubleConv_up(768, 512)
        # self.conv6 = DoubleConv_up(768, 512)

        # self.up7 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(512, 512, 2, stride=2)

        self.conv7 = DoubleConv_up(640, 256)

        self.up8 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv8 = DoubleConv_up(320, 128)

        self.up9 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        # self.conv9 = DoubleConv_up(160, 64)

        self.conv9 = DoubleConv_up(160, 128)

        self.conv10 = DoubleConv_up(128, 64)

        self.eca = FcaLayer(64)

        self.conv11 = DoubleConv_up(128, 64)

        self.conv12 = nn.Conv2d(64, 2, 1)

        # self.conv10 = nn.Conv2d(64, 2, 1)

        self.out1 = nn.Conv2d(96, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.out11 = nn.Conv2d(96, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.out2 = nn.Conv2d(192, 192, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.bottom = nn.Conv2d(672, 672, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.interpolate(x, size=(img_shape // 2, img_shape // 2), mode='bilinear')
        x_3 = F.interpolate(x, size=(img_shape // 4, img_shape // 4), mode='bilinear')
        x_4 = F.interpolate(x, size=(img_shape // 8, img_shape // 8), mode='bilinear')

        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        out1 = torch.cat([self.down1(x_2), p1], dim=1)
        c2 = self.conv2(out1)
        p2 = self.pool2(c2)

        out2 = torch.cat([self.down2(x_3), p2], dim=1)
        c3 = self.conv3(out2)
        p3 = self.pool3(c3)

        out3 = torch.cat([self.down3(x_4), p3], dim=1)
        c4 = self.conv4(out3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        out1 = self.out11(self.out1(out1))
        out2 = self.out2(out2)

        bottom = torch.cat([out1, out2, out3], dim=1)
        bottom = self.bottom(bottom)

        bottom_out = torch.cat([bottom, c5], dim=1)

        up_6 = self.up6(bottom_out)

        merge6 = torch.cat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加

        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)

        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)

        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)

        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)

        c11 = self.eca(c10)

        merge10 = torch.cat([c10, c11], dim=1)

        c12 = self.conv11(merge10)

        #print(c10.shape)
        #print(c11.shape)
        c12 = self.conv12(c12)

        #out = nn.Sigmoid()(c12)  # 化成(0~1)区间
        return c12

       
#多频注意力
import math
import torch
import torch.nn as nn


def get_ld_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def get_dct_weights(width, height, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, height)

    # split channel for multi-spectral attention
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                val = get_ld_dct(t_x, u_x, width) * get_ld_dct(t_y, v_y, height)
                dct_weights[:, i * c_part: (i+1) * c_part, t_x, t_y] = val

    return dct_weights


class FcaLayer(nn.Module):
    def __init__(self, channels,height, channel, fidx_u, fidx_v, reduction=16):
        super(FcaLayer, self).__init__()
        self.register_buffer("precomputed_dct_weights", get_dct_weights(height, channel, fidx_u, fidx_v))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,_,_ = x.size()
        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        y = self.fc(y).view(n,c,1,1)
        return x * y.expand_as(x)
    

# 注意力模块
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    '''def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()

        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.register_buffer('pre_computed_dct_weights',get_dct_weights(width,height,channel ,fidx_u,fidx_v))
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

      

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        #n,c,_,_ = x.size()

        # feature descriptor on the global spatial information
        #y = self.avg_pool(x)
        y=torch.sum(x*self.pre_computed_dct_weights,dim=(2,3))

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)'''
    def __init__(self, channels, reduction=16):
        super(eca_layer, self).__init__()
        self.register_buffer("precomputed_dct_weights", get_dct_weights(...))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        y = self.fc(y).view(n,c,1,1)
        return x * y.expand_as(x)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))