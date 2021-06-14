import torch
import torch.nn as nn
import torchvision.models as models
import copy

from model.HolisticAttention import HA
#from ResNet import B2_ResNet
from torchkeras import summary

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class CPD_MobileNet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32):
        super(CPD_MobileNet, self).__init__()
        self.resnet = models.mobilenet_v3_large(pretrained=True)
        self.br2_7 = copy.deepcopy(self.resnet.features[7])
        self.br2_8 = copy.deepcopy(self.resnet.features[8])
        self.br2_9 = copy.deepcopy(self.resnet.features[9])
        self.br2_10 = copy.deepcopy(self.resnet.features[10])
        self.br2_11 = copy.deepcopy(self.resnet.features[11])
        self.br2_12 = copy.deepcopy(self.resnet.features[12])

        self.br2_13 = copy.deepcopy(self.resnet.features[13])
        self.br2_14 = copy.deepcopy(self.resnet.features[14])
        self.br2_15 = copy.deepcopy(self.resnet.features[15])
        self.br2_16 = copy.deepcopy(self.resnet.features[16])
        # print(self.resnet)
        # print(summary(self.resnet, (3, 256, 256)))
        # input('wait')
        self.rfb2_1 = RFB(40, channel)
        self.rfb3_1 = RFB(112, channel)
        self.rfb4_1 = RFB(960, channel)
        self.agg1 = aggregation(channel)

        self.rfb2_2 = RFB(40, channel)
        self.rfb3_2 = RFB(112, channel)
        self.rfb4_2 = RFB(960, channel)
        self.agg2 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.HA = HA()
        # if self.training:
        #     self.initialize_weights()

    def forward(self, x):
        x = self.resnet.features[0](x)# 128x128
        x = self.resnet.features[1](x)

        x1 = self.resnet.features[2](x)
        x1 = self.resnet.features[3](x1)

        x2 = self.resnet.features[4](x1)
        x2 = self.resnet.features[5](x2)
        x2 = self.resnet.features[6](x2)

        # x = self.resnet.bn1(x)
        # x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)
        #x1 = self.resnet.features[2:4](x)  # 256 x 64 x 64
        #x2 = self.resnet.features[4:7](x1)  # 512 x 32 x 32

        x2_1 = x2
        x3_1 = self.resnet.features[7](x2_1)  # 1024 x 16 x 16
        x3_1 = self.resnet.features[8](x3_1)  # 1024 x 16 x 16
        x3_1 = self.resnet.features[9](x3_1)  # 1024 x 16 x 16
        x3_1 = self.resnet.features[10](x3_1)  # 1024 x 16 x 16
        x3_1 = self.resnet.features[11](x3_1)  # 1024 x 16 x 16
        x3_1 = self.resnet.features[12](x3_1)  # 1024 x 16 x 16

        x4_1 = self.resnet.features[13](x3_1)  # 2048 x 8 x 8
        x4_1 = self.resnet.features[14](x4_1)  # 2048 x 8 x 8
        x4_1 = self.resnet.features[15](x4_1)  # 2048 x 8 x 8
        x4_1 = self.resnet.features[16](x4_1)  # 2048 x 8 x 8

        x2_1 = self.rfb2_1(x2_1)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        attention_map = self.agg1(x4_1, x3_1, x2_1)

        x2_2 = self.HA(attention_map.sigmoid(), x2)
        x3_2 = self.br2_7(x2_2)  # 1024 x 16 x 16
        x3_2 = self.br2_8(x3_2)
        x3_2 = self.br2_9(x3_2)
        x3_2 = self.br2_10(x3_2)
        x3_2 = self.br2_11(x3_2)
        x3_2 = self.br2_12(x3_2)


        x4_2 = self.br2_13(x3_2)  # 2048 x 8 x 8
        x4_2 = self.br2_14(x4_2)
        x4_2 = self.br2_15(x4_2)
        x4_2 = self.br2_16(x4_2)

        x2_2 = self.rfb2_2(x2_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        detection_map = self.agg2(x4_2, x3_2, x2_2)

        return self.upsample(attention_map), self.upsample(detection_map)#

    # def initialize_weights(self):
    #     res50 = models.resnet50(pretrained=True)
    #     pretrained_dict = res50.state_dict()
    #     all_params = {}
    #     for k, v in self.resnet.state_dict().items():
    #         if k in pretrained_dict.keys():
    #             v = pretrained_dict[k]
    #             all_params[k] = v
    #         elif '_1' in k:
    #             name = k.split('_1')[0] + k.split('_1')[1]
    #             v = pretrained_dict[name]
    #             all_params[k] = v
    #         elif '_2' in k:
    #             name = k.split('_2')[0] + k.split('_2')[1]
    #             v = pretrained_dict[name]
    #             all_params[k] = v
    #     assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
    #     self.resnet.load_state_dict(all_params)


class CPD_MobileNet_Single(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32):
        super(CPD_MobileNet_Single, self).__init__()
        self.resnet = models.mobilenet_v3_large(pretrained=True)
        self.br2_7 = copy.deepcopy(self.resnet.features[7])
        self.br2_8 = copy.deepcopy(self.resnet.features[8])
        self.br2_9 = copy.deepcopy(self.resnet.features[9])
        self.br2_10 = copy.deepcopy(self.resnet.features[10])
        self.br2_11 = copy.deepcopy(self.resnet.features[11])
        self.br2_12 = copy.deepcopy(self.resnet.features[12])

        self.br2_13 = copy.deepcopy(self.resnet.features[13])
        self.br2_14 = copy.deepcopy(self.resnet.features[14])
        self.br2_15 = copy.deepcopy(self.resnet.features[15])
        self.br2_16 = copy.deepcopy(self.resnet.features[16])
        # print(self.resnet)
        # print(summary(self.resnet, (3, 256, 256)))
        # input('wait')
        self.rfb2_1 = RFB(40, channel)
        self.rfb3_1 = RFB(112, channel)
        self.rfb4_1 = RFB(960, channel)
        self.agg1 = aggregation(channel)

        self.rfb2_2 = RFB(40, channel)
        self.rfb3_2 = RFB(112, channel)
        self.rfb4_2 = RFB(960, channel)
        self.agg2 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.HA = HA()
        # if self.training:
        #     self.initialize_weights()

    def forward(self, x):
        x = self.resnet.features[0](x)# 128x128
        x = self.resnet.features[1](x)

        x1 = self.resnet.features[2](x)
        x1 = self.resnet.features[3](x1)

        x2 = self.resnet.features[4](x1)
        x2 = self.resnet.features[5](x2)
        x2 = self.resnet.features[6](x2)

        # x = self.resnet.bn1(x)
        # x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)
        #x1 = self.resnet.features[2:4](x)  # 256 x 64 x 64
        #x2 = self.resnet.features[4:7](x1)  # 512 x 32 x 32

        x2_1 = x2
        x3_1 = self.resnet.features[7](x2_1)  # 1024 x 16 x 16
        x3_1 = self.resnet.features[8](x3_1)  # 1024 x 16 x 16
        x3_1 = self.resnet.features[9](x3_1)  # 1024 x 16 x 16
        x3_1 = self.resnet.features[10](x3_1)  # 1024 x 16 x 16
        x3_1 = self.resnet.features[11](x3_1)  # 1024 x 16 x 16
        x3_1 = self.resnet.features[12](x3_1)  # 1024 x 16 x 16

        x4_1 = self.resnet.features[13](x3_1)  # 2048 x 8 x 8
        x4_1 = self.resnet.features[14](x4_1)  # 2048 x 8 x 8
        x4_1 = self.resnet.features[15](x4_1)  # 2048 x 8 x 8
        x4_1 = self.resnet.features[16](x4_1)  # 2048 x 8 x 8

        x2_1 = self.rfb2_1(x2_1)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        attention_map = self.agg1(x4_1, x3_1, x2_1)

        x2_2 = self.HA(attention_map.sigmoid(), x2)
        x3_2 = self.br2_7(x2_2)  # 1024 x 16 x 16
        x3_2 = self.br2_8(x3_2)
        x3_2 = self.br2_9(x3_2)
        x3_2 = self.br2_10(x3_2)
        x3_2 = self.br2_11(x3_2)
        x3_2 = self.br2_12(x3_2)


        x4_2 = self.br2_13(x3_2)  # 2048 x 8 x 8
        x4_2 = self.br2_14(x4_2)
        x4_2 = self.br2_15(x4_2)
        x4_2 = self.br2_16(x4_2)

        x2_2 = self.rfb2_2(x2_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        detection_map = self.agg2(x4_2, x3_2, x2_2)

        return self.upsample(detection_map)#