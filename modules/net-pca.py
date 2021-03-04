import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.ops import ModulatedDeformConv2d

channel_num = [32, 32, 32, 32, 32, 32]


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class DownSp_Block(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(DownSp_Block, self).__init__()
        self.one = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, 1, 1),
            nn.BatchNorm2d(out_chan),

            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, 2, 1),
            nn.BatchNorm2d(out_chan),
        )

        self.two = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, 2, 1),
            nn.BatchNorm2d(out_chan),

            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, 1, 1),
            nn.BatchNorm2d(out_chan),
        )

        self.three = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, return_indices=True)

        self.fuse = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, 1, 1),
            nn.BatchNorm2d(out_chan),
        )

    def forward(self, x):
        x = F.relu(x, inplace=True)

        feat_one = self.one(x)
        feat_two = self.two(x)
        feat_three, indices = self.three(x)

        feat = feat_one + feat_two + feat_three
        feat = self.fuse(feat)
        return feat, indices


class UpSp_Block(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(UpSp_Block, self).__init__()

        self.one = nn.Sequential(
            nn.Conv2d(in_chan * 2, in_chan, 3, 1, 1),
            nn.BatchNorm2d(in_chan),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_chan, in_chan, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_chan),
        )

        self.two = nn.Sequential(
            nn.ConvTranspose2d(in_chan * 2, in_chan, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_chan),

            nn.ReLU(inplace=True),
            nn.Conv2d(in_chan, out_chan, 3, 1, 1),
            nn.BatchNorm2d(out_chan),
        )

        self.three_1 = nn.Sequential(
            nn.Conv2d(in_chan * 2, out_chan, 3, 1, 1),
            nn.BatchNorm2d(out_chan),
        )
        self.three_2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

        self.fuse = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, 1, 1),
            nn.BatchNorm2d(out_chan),
        )

    def forward(self, x, indices):
        x = F.relu(x, inplace=True)

        feat_one = self.one(x)
        feat_two = self.two(x)
        feat_three = self.three_2(self.three_1(x), indices)

        feat = feat_one + feat_two + feat_three
        feat = self.fuse(feat)
        return feat


class DownIBS_Block(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(DownIBS_Block, self).__init__()
        mid_chan = out_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, out_chan, 3, stride=1)

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                out_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=out_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )

        self.dwconv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.conv1x1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chan),

            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_chan),
        )

    def forward(self, x):
        x = F.relu(x, inplace=True)
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv1x1(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        return feat


class UpIBS_Block(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(UpIBS_Block, self).__init__()
        mid_chan = in_chan * exp_ratio

        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )

        self.dwconv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.conv1x1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_chan, in_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(in_chan),
        )

        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(in_chan, in_chan, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(in_chan),
        )

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_chan, in_chan, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_chan),

            nn.ReLU(inplace=True),
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chan),
        )

    def forward(self, x):
        x = F.relu(x, inplace=True)
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv1x1(feat)
        feat = self.conv_up(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        return feat


class IBFE_Block(nn.Module):

    def __init__(self, chan, exp_ratio=6):
        super(IBFE_Block, self).__init__()
        mid_channels = chan * exp_ratio
        self.conv1 = ConvBNReLU(chan, chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                chan, mid_channels, kernel_size=3, stride=1,
                padding=1, groups=chan, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(chan),
        )

    def forward(self, x):
        x = F.relu(x, inplace=True)
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        return feat


class DGF_Block(nn.Module):
    def __init__(self, channels):
        super(DGF_Block, self).__init__()

        # -----------
        # dcn
        self.conv_offset = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, 3 * 3 * 3, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        )
        self.conv_deform = ModulatedDeformConv2d(channels, channels, 3, 1, 1)
        self.conv_offset[-1].weight.data.zero_()
        self.conv_offset[-1].bias.data.zero_()
        # -----------

        self.conv_out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(channels, 1, 3, 1, 1)
        )

    def forward(self, feat_rgb, feat_depth):
        # deform
        out = self.conv_offset(feat_rgb)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_np = offset.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        np.save("offset2.npy", offset_np)
        np.save("mask2.npy", mask_np)


        deform_feat = self.conv_deform(feat_depth, offset, mask)

        feat_depth = feat_depth + deform_feat

        feat_fusion = torch.cat((feat_depth, feat_rgb), dim=1)

        depth_out = self.conv_out(feat_fusion)

        return depth_out


class Aggregation(nn.Module):
    def __init__(self, chan):
        super(Aggregation, self).__init__()
        self.flow = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(chan * 2, chan, 3, 1, 1),
            nn.BatchNorm2d(chan)
        )

    def forward(self, a, b):
        a = torch.cat((a, b), dim=1)
        return self.flow(a)


class DepthEncoder(nn.Module):
    def __init__(self, in_channels, channel_list):
        super(DepthEncoder, self).__init__()
        self.init = nn.Sequential(
            nn.Conv2d(in_channels, channel_list[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),

            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], channel_list[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
        )

        self.enc1 = DownSp_Block(channel_list[0], channel_list[1])

        self.enc2 = DownSp_Block(channel_list[1], channel_list[2])

        self.enc3 = nn.Sequential(
            DownIBS_Block(channel_list[2], channel_list[3]),
            IBFE_Block(channel_list[3]),
            IBFE_Block(channel_list[3]),
        )

        self.conv1x1_x6 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[0], 1, 1, 0),
            nn.BatchNorm2d(channel_list[0])
        )

        self.conv1x1_x5 = nn.Sequential(
            nn.Conv2d(channel_list[2], channel_list[1], 1, 1, 0),
            nn.BatchNorm2d(channel_list[1])
        )

        self.conv1x1_x4 = nn.Sequential(
            nn.Conv2d(channel_list[3], channel_list[2], 1, 1, 0),
            nn.BatchNorm2d(channel_list[2])
        )

        self.conv1x1_x3 = nn.Sequential(
            nn.Conv2d(channel_list[-1], channel_list[3], 1, 1, 0),
            nn.BatchNorm2d(channel_list[3])
        )

    def forward(self, input, scale=2, pre_x3=None, pre_x4=None, pre_x5=None, pre_x6=None):

        x0 = self.init(input)  # 1/1 input size
        if pre_x6 is not None:
            pre_x6 = F.interpolate(pre_x6, scale_factor=scale, mode='bilinear', align_corners=True)
            x0 = x0 + self.conv1x1_x6(pre_x6)

        x1, indices1 = self.enc1(x0)  # 1/2 input size
        if pre_x5 is not None:
            pre_x5 = F.interpolate(pre_x5, scale_factor=scale, mode='bilinear', align_corners=True)
            x1 = x1 + self.conv1x1_x5(pre_x5)

        x2, indices2 = self.enc2(x1)  # 1/4 input size
        if pre_x4 is not None:
            pre_x4 = F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=True)
            x2 = x2 + self.conv1x1_x4(pre_x4)

        x3 = self.enc3(x2)  # 1/8 input size
        if pre_x3 is not None:  # newly added skip connection
            pre_x3 = F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=True)
            x3 = x3 + self.conv1x1_x3(pre_x3)

        return (x0, x1, x2, x3), indices2, indices1


class DepthDecoder(nn.Module):
    def __init__(self, channel_list):
        super(DepthDecoder, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel_list[0] * 2, channel_list[0], 1, 1, 0),
            nn.BatchNorm2d(channel_list[0])
        )

        self.dec3 = UpIBS_Block(channel_list[0], channel_list[1])

        self.aggregation3 = Aggregation(channel_list[1])

        self.dec2_1 = UpSp_Block(channel_list[1], channel_list[2])
        self.enc2_1 = DownSp_Block(channel_list[2], channel_list[1])
        self.aggregation21 = Aggregation(channel_list[1])

        self.dec2_2 = UpSp_Block(channel_list[1], channel_list[2])

        self.aggregation2_1 = Aggregation(channel_list[2])
        self.aggregation2_2 = Aggregation(channel_list[2])

        self.dec1_1 = UpSp_Block(channel_list[2], channel_list[3])
        self.enc1_1 = DownSp_Block(channel_list[3], channel_list[2])
        self.aggregation11 = Aggregation(channel_list[2])

        self.dec1_2 = UpSp_Block(channel_list[2], channel_list[3])
        self.enc1_2 = DownSp_Block(channel_list[3], channel_list[2])
        self.aggregation12 = Aggregation(channel_list[2])

        self.dec1_3 = UpSp_Block(channel_list[2], channel_list[3])

        self.aggregation1_1 = Aggregation(channel_list[3])
        self.aggregation1_2 = Aggregation(channel_list[3])
        self.aggregation1_3 = Aggregation(channel_list[3])

        self.fuse = DGF_Block(channel_list[3])

    def forward(self, enc_sd, enc_rgb, indices2, indices1):
        return_list = []

        x2 = torch.cat((enc_sd[3], enc_rgb[3]), dim=1)
        x2 = self.conv1x1(x2)

        return_list.append(x2)

        x3 = self.dec3(x2)
        x3 = self.aggregation3(enc_sd[2], x3)
        return_list.append(x3)

        x4_1 = self.dec2_1(
            torch.cat((enc_sd[2], enc_rgb[2]), dim=1),
            indices2
        )
        x4_1 = self.aggregation2_1(enc_sd[1], x4_1)
        x3_, indices2_1 = self.enc2_1(x4_1)
        x3 = self.aggregation21(x3, x3_)

        x4_2 = self.dec2_2(
            torch.cat((x3, enc_rgb[2]), dim=1),
            indices2_1
        )
        x4_2 = self.aggregation2_2(x4_1, x4_2)
        return_list.append(x4_2)

        x5_1 = self.dec1_1(
            torch.cat((enc_sd[1], enc_rgb[1]), dim=1),
            indices1
        )
        x5_1 = self.aggregation1_1(enc_sd[0], x5_1)
        x4_1_, indices1_1 = self.enc1_1(x5_1)
        x4_1 = self.aggregation11(x4_1, x4_1_)

        x5_2 = self.dec1_2(
            torch.cat((x4_1, enc_rgb[1]), dim=1),
            indices1_1
        )
        x5_2 = self.aggregation1_2(x5_1, x5_2)
        x4_2_, indices1_2 = self.enc1_2(x5_2)
        x4_2 = self.aggregation12(x4_2, x4_2_)

        x5_3 = self.dec1_3(
            torch.cat((x4_2, enc_rgb[1]), dim=1),
            indices1_2
        )
        x5_3 = self.aggregation1_3(x5_2, x5_3)
        return_list.append(x5_3)

        output_d = self.fuse(x5_3, enc_rgb[0])
        return_list.append(output_d)

        return return_list


class RGBEncoder(nn.Module):
    def __init__(self, in_channels):
        super(RGBEncoder, self).__init__()

        self.init = nn.Sequential(
            nn.Conv2d(in_channels, channel_num[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num[0]),

            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[0], channel_num[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num[0]),
        )

        self.enc1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[0], channel_num[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(channel_num[1]),

            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[1], channel_num[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num[1]),
        )

        self.enc2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[1], channel_num[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(channel_num[2]),

            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[2], channel_num[2], 3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num[2]),
        )

        self.enc3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[2], channel_num[3], 3, stride=2, padding=1),
            nn.BatchNorm2d(channel_num[3]),

            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[3], channel_num[3], 3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num[3]),
        )

        self.enc4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[3], channel_num[4], 3, stride=2, padding=1),
            nn.BatchNorm2d(channel_num[4]),

            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[4], channel_num[4], 3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num[4]),
        )

        self.enc5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[4], channel_num[5], 3, stride=2, padding=1),
            nn.BatchNorm2d(channel_num[5]),

            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num[5], channel_num[5], 3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num[5]),
        )

    def forward(self, input):
        x0 = self.init(input)  # 1/1 feat size
        x1 = self.enc1(x0)  # 1/2 feat size
        x2 = self.enc2(x1)  # 1/4 feat size
        x3 = self.enc3(x2)  # 1/8 feat size
        x4 = self.enc4(x3)  # 1/16 feat size
        x5 = self.enc5(x4)  # 1/32 feat size

        return x0, x1, x2, x3, x4, x5


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        self.rgb_encoder = RGBEncoder(3)

        self.depth_encoder1 = DepthEncoder(2, channel_num[2:6])
        self.depth_decoder1 = DepthDecoder(channel_num[2:6][::-1])

        self.depth_encoder2 = DepthEncoder(3, channel_num[1:6])
        self.depth_decoder2 = DepthDecoder(channel_num[1:5][::-1])

        self.depth_encoder3 = DepthEncoder(3, channel_num[0:5])
        self.depth_decoder3 = DepthDecoder(channel_num[0:4][::-1])

    def forward(self, input_d, input_rgb):
        C11 = (input_d > 0).float()

        enc_rgb = self.rgb_encoder(input_rgb)

        # for the 1/4 res
        input_d18 = F.avg_pool2d(input_d, 8, 8) / (F.avg_pool2d(C11, 8, 8) + 0.0001)
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C11, 4, 4) + 0.0001)
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C11, 2, 2) + 0.0001)

        C12 = (input_d12 > 0).float()
        C14 = (input_d14 > 0).float()

        temp_d14 = F.interpolate(input_d18, scale_factor=2)
        mask14 = torch.logical_and(input_d14 == 0, temp_d14 != 0).float()
        weight_14 = (input_d14 != 0).float() + F.interpolate(F.avg_pool2d(C14, 2, 2), scale_factor=2) * mask14
        input_d14 += temp_d14 * mask14

        temp_d12 = F.interpolate(input_d14, scale_factor=2)
        mask12 = torch.logical_and(input_d12 == 0, temp_d12 != 0).float()
        weight_12 = (input_d12 != 0).float() + \
                    F.interpolate(F.avg_pool2d(C12, 2, 2), scale_factor=2) * mask12.float() + \
                    F.interpolate(weight_14, scale_factor=2) * (1 - F.interpolate(C14, scale_factor=2)) * 0.25
        input_d12 += temp_d12 * mask12

        temp_d11 = F.interpolate(input_d12, scale_factor=2)
        mask11 = torch.logical_and(input_d == 0, temp_d11 != 0).float()
        weight_11 = (input_d != 0).float() + \
                    F.interpolate(F.avg_pool2d(C11, 2, 2), scale_factor=2) * mask11.float() + \
                    F.interpolate(weight_12, scale_factor=2) * (1 - F.interpolate(C12, scale_factor=2)) * 0.25

        input_d += temp_d11 * mask11

        enc_d14, indices2, indices1 = self.depth_encoder1(
            torch.cat((input_d14, weight_14), dim=1),
        )
        dcd_d14 = self.depth_decoder1(enc_d14, enc_rgb[2:6], indices2, indices1)

        # for the 1/2 res
        predict_d12 = F.interpolate(dcd_d14[4], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, weight_12, predict_d12), 1)

        enc_d12, indices2, indices1 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2], dcd_d14[3])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_rgb[1:5], indices2, indices1)

        # for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[4] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, weight_11, predict_d11), 1)

        enc_d11, indices2, indices1 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2], dcd_d12[3])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_rgb[0:4], indices2, indices1)

        output_d11 = dcd_d11[4] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[4], scale_factor=4, mode='bilinear', align_corners=True)

        return output_d11, output_d12, output_d14,


if __name__ == '__main__':
    rgb = torch.randn([1, 3, 352, 1216]).cuda()
    sparse_depth = torch.randn([1, 1, 352, 1216]).cuda()
    model = network().cuda()
    output = model(sparse_depth, rgb)
