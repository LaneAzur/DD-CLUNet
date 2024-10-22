'''
VoxelMorph

Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Yicheng Chen, Shengxiang Ji, Yuelin Xin
'''
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.normal import Normal

from utils.utils import SpatialTransformer

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()

        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):

            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool3d(x,(anchor.shape[2],anchor.shape[3], anchor.shape[4]))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(anchor.shape[2],anchor.shape[3], anchor.shape[4]),
                                      mode='trilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Conv_Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv_Block, self).__init__()
        self.conv = nn.Sequential(
            # nn.BatchNorm3d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class dens_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(dens_block, self).__init__()#这三个相同吗？？？？
        self.conv1 = Conv_Block(ch_in,ch_out)
        self.conv2 = Conv_Block(ch_out+ch_in, ch_out)
        self.conv3 = Conv_Block(ch_out*2 + ch_in, ch_out)
    def forward(self,input_tensor):
        x1 = self.conv1(input_tensor)
        # print('x1',x1.shape)
        add1 = torch.cat([x1,input_tensor],dim=1)
        x2 = self.conv2(add1)
        # print('x2',x2.shape)
        add2 =torch.cat([x1, input_tensor,x2], dim=1)
        x3 = self.conv3(add2)
        # print('x3',x3.shape)
        return x3

class Unet(nn.Module):
    def __init__(self, dim = 3,nb_features=None):
        super(Unet, self).__init__()

        self.dim = dim
        self.enc = nn.ModuleList()
        nb_features = ((8, 32, 32, 32), (32, 32, 32, 32, 32, 8, 8))
        self.enc_nf, self.dec_nf = nb_features
        enc_nf = self.enc_nf
        dec_nf = self.dec_nf
        for i in range(len(enc_nf)):
            prev_nf = 1 if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 4, 2))
            # 0 2-16
            # 1 16-32
            # 2 32-32
            # 3 32-32
        # print(self.enc)
        # Decoder functions
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_x1 = torch.nn.Linear(32, 32)
        self.linear_y1 = torch.nn.Linear(32, 32)

        self.ca_1 = ChannelAttention(32)
        # self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(32)
        # self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(8)
        # self.sa_3 = SpatialAttention()
        self.Translayer_0 = BasicConv3d(32, 32, 1)
        self.Translayer_1 = BasicConv3d(32, 32, 1)
        self.Translayer_2 = BasicConv3d(32, 32, 1)
        self.Translayer_3 = BasicConv3d(8, 32, 1)

        self.SDI = SDI(32)

        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(self.conv_block(dim,64,32))
        self.bottleneck.append(self.conv_block(dim,96,64))
        self.bottleneck.append(self.conv_block(dim, 96, 64))
        self.bottleneck.append(self.conv_block(dim, 48, 40))
        self.bottleneck.append(self.conv_block(dim, 34, 34))


        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1 32-32
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2 32-32
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3 64-32
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4 32+16=48 32
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4]))  # 5 32-32

        self.dec.append(self.conv_block(dim, dec_nf[4] + 2, dec_nf[5]))  # 32+2 -> 16

        self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6])  # 16 -> 16

        self.zMaxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.zUp6 = up_conv(32,32)
        self.zadd6 = torch.cat

        self.zup6 = dens_block(32+32+32+32,32)

        self.zUp7 = up_conv(32,32)
        self.zadd7 = torch.cat
        self.zup7 = dens_block(32+32+32+32,32)

        self.zUp8 = up_conv(32,32)
        self.zadd8 = torch.cat

        self.zup8 = dens_block(32+32+8+8,32)

        self.zUp9 = up_conv(32,32)
        self.zadd9 = torch.cat
        self.zup9 = dens_block(32+8+2,8)

        self.zconv10_1 = nn.Conv3d(8,8,3,1,1)
        self.zrelu = nn.ReLU(inplace=True)
        self.zconv10_2 = nn.Conv3d(8, 8, 3, 1, 1)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))  # 通过变量控制神经网络的函数名称 conv3d
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        # 膨胀卷积即反卷积
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x,t):
        ## -------------Encoder-------------
        # Get encoder activations

        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        #

        t_enc = [t]
        for i, l in enumerate(self.enc):
            t = l(t_enc[-1])
            t_enc.append(t)

        f0x = self.ca_1(x_enc[4]) * x_enc[4]
        # f1y = self.sa_1(encode_block1_y) * f1y
        f0x = self.Translayer_0(f0x)
        f1x = self.ca_1(x_enc[3]) * x_enc[3]
        # f1x = self.sa_1(encode_block1_x) * f1x
        f1x = self.Translayer_1(f1x)
        f2x = self.ca_2(x_enc[2]) * x_enc[2]
        # f2x = self.sa_2(encode_block2_x) * f2x
        f2x = self.Translayer_2(f2x)
        f3x = self.ca_3(x_enc[1]) * x_enc[1]
        # f3x = self.sa_3(encode_block3_x) * f3x
        f3x = self.Translayer_3(f3x)
        f_x = self.SDI([f0x,f1x, f2x, f3x], f3x)
        f_x = self.avgpool(f_x)
        f_x = f_x.squeeze()
        f_x = self.linear_x1(f_x)
        f_x = f_x / f_x.norm(dim=-1, keepdim=True)

        f0y = self.ca_1(t_enc[4]) * t_enc[4]
        # f1y = self.sa_1(encode_block1_y) * f1y
        f0y = self.Translayer_0(f0y)
        f1y = self.ca_1(t_enc[3]) * t_enc[3]
        # f1y = self.sa_1(encode_block1_y) * f1y
        f1y = self.Translayer_1(f1y)
        f2y = self.ca_2(t_enc[2]) * t_enc[2]
        # f2y = self.sa_2(encode_block2_y) * f2y
        f2y = self.Translayer_2(f2y)
        f3y = self.ca_3(t_enc[1]) * t_enc[1]
        # f3y = self.sa_3(encode_block3_y) * f3y
        f3y = self.Translayer_3(f3y)
        f_y = self.SDI([f0y,f1y, f2y, f3y], f3y)
        print('f_y',f_y.shape)
        f_y = self.avgpool(f_y)
        f_y = f_y.squeeze()
        f_y = self.linear_y1(f_y)
        f_y = f_y / f_y.norm(dim=-1, keepdim=True)

        y = torch.cat((x_enc[4],t_enc[4]),1)
        y = self.bottleneck[0](y)
        y = self.dec[0](y)
        y1 = y

        y = self.upsample(y)

        y = self.crop_and_concat(y, torch.cat((x_enc[3], t_enc[3]), 1))
        # y = torch.cat([y, x_enc[3]], dim=1)
        y = self.bottleneck[1](y)
        y = self.dec[1](y)
        y2 = y

        y = self.upsample(y)
        y = self.crop_and_concat(y, torch.cat((x_enc[2], t_enc[2]), 1))
        # y = torch.cat([y, x_enc[2]], dim=1)
        y = self.bottleneck[2](y)
        y = self.dec[2](y)
        y3 = y

        y = self.upsample(y)
        y = self.crop_and_concat(y, torch.cat((x_enc[1], t_enc[1]), 1))
        y = self.bottleneck[3](y)
        # y = torch.cat([y, x_enc[1]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y4 = y

        y = self.dec[4](y)
        # print(self.dec[4])
        y5 = y

        # Upsample to full res, concatenate and conv

        y = self.upsample(y)
        # print('x_enc[0]',x_enc[0].shape)
        y = self.crop_and_concat(y, torch.cat((x_enc[0], t_enc[0]), 1))
        y = self.bottleneck[4](y)
        # y = torch.cat([y, x_enc[0]], dim=1)

        y = self.dec[5](y)
        # print(self.dec[5])
        y6 = y

        # Extra conv for vm2
        y = self.vm2_conv(y)
        y7 = y

        down1 = y6
        down2 = y4
        down3 = y3
        down4 = y2
        conv5 = y1

        up6 = self.zUp6(conv5)
        add6 = self.zadd6([x_enc[3],t_enc[3],down4,up6],dim=1)

        up6 = self.zup6(add6)

        up7 = self.zUp7(up6)
        add7 = self.zadd7([x_enc[2],t_enc[2], down3,up7],dim=1)

        up7 = self.zup7(add7)

        up8 = self.zUp8(up7)
        add8 = self.zadd8([x_enc[1],t_enc[1], down2,up8],dim=1)

        up8 = self.zup8(add8)

        up9 = self.zUp9(up8)
        add9 = self.zadd9([x_enc[0],t_enc[0],down1,up9],dim=1)

        up9 = self.zup9(add9)


        conv10 = self.zconv10_1(up9)
        conv10 = self.zrelu(conv10)
        conv10 = self.zconv10_2(conv10)

        return conv10,f_x,f_y


class VoxelMorph(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            3,
            nb_features=nb_unet_features,
            # nb_levels=nb_unet_levels,
            # feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor,y):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        flow,f_x,f_y = self.unet_model(x,y)

        # transform into flow field
        flow_field = self.flow(flow)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(x, pos_flow)

        return y_source, pos_flow,f_x,f_y
