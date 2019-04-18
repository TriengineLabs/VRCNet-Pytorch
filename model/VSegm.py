import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg
from icecream import ic

class UpBlock(nn.Module):
    def __init__(self, input_channels, output_channels, include_batch_norm=True):
        super(UpBlock, self).__init__()
        
        if include_batch_norm:
            self.up = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=True),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=True),
                nn.ReLU()
            )

    def forward(self, x):
        return self.up(x)

class UpBlockBig(nn.Module):
    def __init__(self, input_channels, output_channels, include_batch_norm=True):
        super(UpBlockBig, self).__init__()
        
        if include_batch_norm:
            self.up = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(),
                nn.Conv2d(input_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else: 
            self.up = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=False),
                nn.ReLU(),
                nn.Conv2d(input_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=False),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=False),
                nn.ReLU()
            )

    def forward(self, x):
        return self.up(x)

class VSegm(nn.Module):
    def __init__(self, output_channels=1):
        super(VSegm, self).__init__()
        
        self.down = vgg.vgg16_bn(pretrained=True).features


        self.first_conv = nn.Conv2d(1, 3, kernel_size=5,
                stride=1, padding=2)

        self.mid_conv = nn.Conv2d(512, 512, kernel_size=5,
                stride=1, padding=2)

        self.up4 = UpBlockBig(512, 512)
        self.up3 = UpBlockBig(512, 256)
        # self.up4 = UpBlock(512, 512)
        # self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)
        # self.up0 = UpBlock(64, 3, include_batch_norm=False)

        self.last_conv = nn.Sequential(
                nn.Conv2d(64, 1,
                kernel_size = 5, stride=1,
                padding=2),
                nn.Sigmoid()
            )

    def forward(self, x):
        d = self.first_conv(x)
        d = nn.ReLU().forward(d)

        

        necessary_outputs = []
        for ii, sub_model in  enumerate(self.down):
            d = sub_model(d)
            # if ii in {5, 12, 22, 32, 42}:
            #     ic(d.shape)
            #     necessary_outputs.append(d)

        mid = self.mid_conv(d)
        mid = nn.ReLU().forward(mid)

        up4 = F.interpolate(mid, size=(14,14))
        # ic('up 4 output shape ', up4.shape)
        up4 = self.up4(up4)

        up3 = F.interpolate(up4, size=(28, 28))
        # ic('up 3 output shape ', up3.shape)
        up3 = self.up3(up3)

        up2 = F.interpolate(up3, size=(56, 56))
        # ic('up 2 output shape ', up2.shape)
        up2 = self.up2(up2)

        up1 = F.interpolate(up2, size=(112, 112))
        # ic('up 1 output shape ', up1.shape)
        up1 = self.up1(up1)

        # up0 = F.interpolate(up1, size=(224, 224))
        # ic('up0 output shape ', up0.shape)
        # up0 = self.up0(up0)

        last = F.interpolate(up1, size=(224, 224))
        last = self.last_conv(last)

        return last

