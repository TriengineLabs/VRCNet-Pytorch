import torch.nn.functional as F
from DeepUNet_utils import *
from icecream import ic


class DeepUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DeepUNet, self).__init__()
        self.enc1 = EncoderLayer(n_channels, 16)
        self.enc2 = EncoderLayer(16, 32)
        self.enc3 = EncoderLayer(32, 64)
        self.enc4 = EncoderLayer(64, 128)
        self.enc5 = EncoderLayer(128, 256)
        self.enc6 = EncoderLayer(256, 512)
        self.dec1 = DecoderLayer(512, 256)
        self.dec2 = DecoderLayer(512, 128)
        self.dec3 = DecoderLayer(256, 64)
        self.dec4 = DecoderLayer(128, 32)
        self.dec5 = DecoderLayer(64, 16)
        self.dec6 = DecoderLayer(32, n_classes)

    def forward(self, x):
        # Extending one dimension that corresponds to 1 channel of the input
        x = x.unsqueeze(1)

        x1 = self.enc1(x)
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
        x = self.dec6(x, concat=False)

        return F.sigmoid(x)
