import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleDownConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DoubleDownConv, self).__init__()

        self.double_conv = nn.Sequential(
                nn.MaxPool2d(3, 2, 1),

                nn.Conv2d(input_channel, output_channel, 5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(),

                nn.Conv2d(output_channel, output_channel, 5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(output_channel),
                nn.ReLU()
                )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class DoubleUpConv(nn.Module):
    def __init__(self, input_channel, output_channel, include_batch_norm=True):
        super(DoubleUpConv, self).__init__()

        if include_batch_norm:
            self.double_conv = nn.Sequential(
                    nn.Conv2d(input_channel, output_channel, 5, stride=1, padding=2, bias=False),
                    nn.BatchNorm2d(output_channel),
                    nn.ReLU(),

                    nn.Conv2d(output_channel, output_channel, 5, stride=1, padding=2, bias=False),
                    nn.BatchNorm2d(output_channel),
                    nn.ReLU()
                    )
        else:
            self.double_conv = nn.Sequential(
                    nn.Conv2d(input_channel, output_channel, 5, stride=1, padding=2, bias=False),
                    nn.ReLU(),

                    nn.Conv2d(output_channel, output_channel, 5, stride=1, padding=2, bias=False),
                    nn.ReLU()
                    )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_classes, filters=32):
        super(Generator, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(1, filters, 5, 1, 2, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),

            nn.Conv2d(filters, filters, 5, 1, 2, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )

        self.downsample1 = DoubleDownConv(filters, filters*2)
        self.downsample2 = DoubleDownConv(filters*2, filters*4)
        self.downsample3 = DoubleDownConv(filters*4, filters*8)
        self.downsample4 = DoubleDownConv(filters*8, filters*16)

        self.mid_conv = nn.Sequential(
            nn.Conv2d(filters*16, filters*8, 1, 1),
            nn.ReLU()
        )

        self.upsample3 = DoubleUpConv(filters*16, filters*4)
        self.upsample2 = DoubleUpConv(filters*8, filters*2)
        self.upsample1 = DoubleUpConv(filters*4, filters)
        self.upsample0 = DoubleUpConv(filters*2, filters)

        self.last_conv = nn.Sequential(
            nn.Conv2d(filters, n_classes, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.unsqueeze(1)

        d0 = self.first_layer(x)

        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)

        #TODO mid_conv place
        u3 = F.interpolate(d4, d3.shape[2:])
        u3 = self.mid_conv(u3)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.upsample3(u3)

        u2 = F.interpolate(u3, d2.shape[2:])
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.upsample2(u2)

        u1 = F.interpolate(u2, d1.shape[2:])
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.upsample1(u1)

        u0 = F.interpolate(u1, d0.shape[2:])
        u0 = torch.cat([u0, d0], dim=1)
        u0 = self.upsample0(u0)

        f = self.last_conv(u0)
        return f
