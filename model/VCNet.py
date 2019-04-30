import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg
from icecream import ic
from torchvision.models.resnet import resnet18

class UpBlock(nn.Module):
    def __init__(self, input_channels, output_channels, include_batch_norm=True):
        super(UpBlock, self).__init__()
        
        if include_batch_norm:
            self.up = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=False),
                # nn.BatchNorm2d(output_channels),
                # nn.ReLU(),
                # nn.Conv2d(output_channels, output_channels, 
                #     kernel_size=5, stride=1,
                #     padding=2, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=True),
                # nn.ReLU(),
                # nn.Conv2d(output_channels, output_channels, 
                #     kernel_size=5, stride=1,
                #     padding=2, bias=True),
                nn.ReLU()
            )

    def forward(self, x):
        return self.up(x)

# class UpBlockBig(nn.Module):
#     def __init__(self, input_channels, output_channels, include_batch_norm=True):
#         super(UpBlockBig, self).__init__()
#         
#         if include_batch_norm:
#             self.up = nn.Sequential(
#                 # nn.Conv2d(input_channels, input_channels, 
#                 #     kernel_size=5, stride=1,
#                 #     padding=2, bias=False),
#                 # nn.BatchNorm2d(input_channels),
#                 # nn.ReLU(),
#                 nn.Conv2d(input_channels, output_channels, 
#                     kernel_size=5, stride=1,
#                     padding=2, bias=False),
#                 nn.BatchNorm2d(output_channels),
#                 nn.ReLU(),
#                 nn.Conv2d(output_channels, output_channels, 
#                     kernel_size=5, stride=1,
#                     padding=2, bias=False),
#                 nn.BatchNorm2d(output_channels),
#                 nn.ReLU()
#             )
#         else: 
#             self.up = nn.Sequential(
#                 # nn.Conv2d(input_channels, input_channels, 
#                 #     kernel_size=5, stride=1,
#                 #     padding=2, bias=False),
#                 # nn.ReLU(),
#                 nn.Conv2d(input_channels, output_channels, 
#                     kernel_size=5, stride=1,
#                     padding=2, bias=False),
#                 nn.ReLU(),
#                 nn.Conv2d(output_channels, output_channels, 
#                     kernel_size=5, stride=1,
#                     padding=2, bias=False),
#                 nn.ReLU()
#             )
# 
#     def forward(self, x):
#         return self.up(x)

class VCNet(nn.Module):
    def __init__(self, output_channels=1):
        super(VCNet, self).__init__()

        self.first_conv = nn.Conv2d(1, 3, kernel_size=5,
                stride=1, padding=2)
 
        #Defining ResNet
        resnet = resnet18(pretrained=True)
        self.resnet_layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.resnet_maxpool1 = resnet.maxpool

        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4
        del resnet

        #Defining VGG
        self.vgg = vgg.vgg16_bn(pretrained=True).features

        # self.worm3 = nn.Sequential(
        #         nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
        #         nn.ReLU())
        self.worm2 = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU())
        self.worm1 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU())
        self.mid_conv = nn.Conv2d(1024, 1024, kernel_size=5,
                stride=1, padding=2)

        self.up4 = UpBlock(1024, 512)
        self.up3 = UpBlock(1152, 256)
        self.up2 = UpBlock(384, 128)
        self.up1 = UpBlock(256, 64)
        self.up0 = UpBlock(64, 32)

        self.last_conv = nn.Sequential(
                nn.Conv2d(32, 1,
                kernel_size = 5, stride=1,
                padding=2),
                nn.Sigmoid()
            )

    def forward(self, x):
        inp = self.first_conv(x)
        inp = nn.ReLU().forward(inp)

        #Passing through VGG
        d = inp
        necessary_shapes = []
        necessary_outputs = []
        for ii, sub_model in  enumerate(self.vgg):
            d = sub_model(d)
            if ii in {5, 12, 22, 32, 42}:
                # ic(d.shape)
                if ii in {12, 22, 32, 42}:
                    necessary_outputs.append(d)
                necessary_shapes.append(d.shape[-2:])

        #Passing through ResNet
        resnetl_0 = self.resnet_layer0(inp)
        resnetl_0_maxpool = self.resnet_maxpool1(resnetl_0)
        resnetl_1 = self.resnet_layer1(resnetl_0_maxpool)
        resnetl_2 = self.resnet_layer2(resnetl_1)
        resnetl_3 = self.resnet_layer3(resnetl_2)
        resnetl = self.resnet_layer4(resnetl_3)

        mid = torch.cat([resnetl[:,:,:32, :5], d], dim=1)

        mid = self.mid_conv(mid)
        mid = nn.ReLU().forward(mid)

        up = F.interpolate(mid, size=(necessary_shapes[4]))
        # up = torch.cat([up, necessary_outputs[3], resnetl_3[:,:,:64, :10]], dim=1)
        up = self.up4(up)

        up = F.interpolate(up, size=(necessary_shapes[3]))
        # ic(necessary_outputs[2].shape)
        # ic(resnetl_2.shape)
        # ic(up.shape)
        up = torch.cat([up, necessary_outputs[2], resnetl_2[:,:,:128, :21]], dim=1)
        up = self.up3(up)

        up = F.interpolate(up, size=(necessary_shapes[2]))
        # ic(necessary_outputs[1].shape)
        # ic(resnetl_1.shape)
        # ic(up.shape)
        vgg_inp = self.worm2(necessary_outputs[1])
        up = torch.cat([up, vgg_inp, resnetl_1[:,:,:256, :43]], dim=1)
        up = self.up2(up)

        up = F.interpolate(up, size=(necessary_shapes[1]))
        # ic(necessary_outputs[0].shape)
        # ic(resnetl_0.shape)
        # ic(up.shape)
        vgg_inp = self.worm1(necessary_outputs[0])
        up = torch.cat([up, vgg_inp, resnetl_0[:,:,:512,:86]], dim=1)
        up = self.up1(up)

        up = F.interpolate(up, size=(necessary_shapes[0]))
        up = self.up0(up)
        up = self.last_conv(up)

        return up

