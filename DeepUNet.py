from DeepUNet_utils import *
import torch
from activation_functions import LeakyRELU

class DeepUNet(nn.Module):
    def __init__(self, n_channels, n_classes, filters=32):
        super(DeepUNet, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(n_channels, filters, 5, 1, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02)
        )

        # self.enc1 = EncoderLayer(n_channels, filters)
        self.enc1 = EncoderLayer(filters, filters*2)
        self.enc2 = EncoderLayer(filters*2, filters*4)
        self.enc3 = EncoderLayer(filters*4, filters*8)
        self.enc4 = EncoderLayer(filters*8, filters*16)
        self.enc5 = EncoderLayer(filters*16, filters*32)
        self.enc6 = EncoderLayer(filters*32, filters*64)
        self.dec1 = DecoderLayer(filters*64, filters*32)
        self.dec2 = DecoderLayer(filters*64, filters*16)
        self.dec3 = DecoderLayer(filters*32, filters*8)
        self.dec4 = DecoderLayer(filters*16, filters*4)
        self.dec5 = DecoderLayer(filters*8, filters*2)
        self.dec6 = DecoderLayer(filters*4, filters)

        self.last_conv = nn.Sequential(
            nn.Conv2d(filters*2, n_classes, 5, 1, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extending one dimension that corresponds to 1 channel of the input
        x = x.unsqueeze(1)

        x0 = self.first_conv(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)

        x = self.dec1(x6, x5)
        x = self.dec2(x, x4)
        x = self.dec3(x, x3)
        x = self.dec4(x, x2)
        x = self.dec5(x, x1)
        x = self.dec6(x, x0)
        x = self.last_conv(x)

        return x
