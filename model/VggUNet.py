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
                nn.ReLU()
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 
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
                    padding=2, bias=False),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, 
                    kernel_size=5, stride=1,
                    padding=2, bias=False),
                nn.ReLU()
            )

    def forward(self, x):
        return self.up(x)

class VggUNet(nn.Module):
    def __init__(self, output_channels=1, freeze_layers=False):
        super(VggUNet, self).__init__()
        
        self.down = vgg.vgg16_bn(pretrained=True).features

        # Freezing layers
        if freeze_layers:
            for submod in self.down[:32]:
                for parameter in submod.parameters():
                    parameter.require_grad = False

        self.first_conv = nn.Conv2d(1, 3, kernel_size=5,
                stride=1, padding=2)

        self.mid_conv = nn.Conv2d(512, 512, kernel_size=5,
                stride=1, padding=2)

        self.up4 = UpBlock(512, 512)
        self.up3 = UpBlock(1024, 256)
        self.up2 = UpBlock(512, 128)
        self.up1 = UpBlock(256, 64)
        self.up0 = UpBlock(64, 32)

        self.last_conv = nn.Sequential(
                nn.Conv2d(32, output_channels,
                kernel_size = 5, stride=1,
                padding=2),
                nn.Sigmoid()
            )

    def forward(self, x):
        d = self.first_conv(x)
        d = nn.ReLU().forward(d)

        necessary_shapes = []
        necessary_outputs = []
        for ii, sub_model in  enumerate(self.down):
            d = sub_model(d)
            if ii in {5, 12, 22, 32, 42}:
                necessary_outputs.append(d)
                necessary_shapes.append(d.shape[-2:])

        mid = self.mid_conv(d)
        mid = nn.ReLU().forward(mid)

        up4 = F.interpolate(mid, size=(necessary_shapes[4]))
        up4 = self.up4(up4)

        up3 = F.interpolate(up4, size=(necessary_shapes[3]))
        up3 = torch.cat([up3, necessary_outputs[3]], dim=1)
        up3 = self.up3(up3)

        up2 = F.interpolate(up3, size=(necessary_shapes[2]))
        up2 = torch.cat([up2, necessary_outputs[2]], dim=1)
        up2 = self.up2(up2)

        up1 = F.interpolate(up2, size=(necessary_shapes[1]))
        up1 = torch.cat([up1, necessary_outputs[1]], dim=1)
        up1 = self.up1(up1)

        last = F.interpolate(up1, size=(necessary_shapes[0]))
        last = self.up0(last)
        last = self.last_conv(last)

        return last

