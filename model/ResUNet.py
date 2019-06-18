import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from torchvision.models.resnet import resnet18

class Generator(nn.Module):
    def __init__(self, output_channels=1, freeze_layers=False):
        super(Generator, self).__init__()

        resnet = resnet18(pretrained=True)

        self.layer_prep = nn.Sequential(nn.Conv2d(1, 3, kernel_size=5,
                                            stride=1, padding=2),
                                        nn.ReLU())

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.maxpool1 = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Freeze layers if needed
        if freeze_layers:
            self.layer1.require_grad = False
            self.layer2.require_grad = False
            self.layer3.require_grad = False
        
        
        self.upsample4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.upsample3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.upsample2 = nn.Sequential(
            nn.Conv2d(384, 192, 3, 1, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )

        self.upsample1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.upsample0 = nn.Sequential(
            nn.Conv2d(192, 96, 3, 1, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.final0 = nn.Sequential(
            nn.Conv2d(96, output_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        l_prep = self.layer_prep(inputs)
        l0 = self.layer0(l_prep)
        l0_maxpool = self.maxpool1(l0)
        l1 = self.layer1(l0_maxpool)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        u4 = self.upsample4(F.interpolate(l4, l3.shape[2:]))

        c3 = torch.cat([l3, u4], dim=1)
        u3 = self.upsample3(F.interpolate(c3, l2.shape[2:]))

        c2 = torch.cat([l2, u3], dim=1)
        u2 = self.upsample2(F.interpolate(c2, l1.shape[2:]))

        c1 = torch.cat([l1, u2], dim=1)
        u1 = self.upsample1(F.interpolate(c1, l0.shape[2:]))

        c0 = torch.cat([l0, u1], dim=1)
        u0 = self.upsample0(F.interpolate(c0, inputs.shape[2:]))
        o0 = self.final0(u0)

        return o0
