import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(EncoderLayer, self).__init__()

        #TODO change Relu to leaky Relu
        #TODO check Conv2d output size
        self.encoding = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 5, stride=2, padding=2),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        x = self.encoding(x)
        return x

class DecoderLayer(nn.Module):
    # TODO change the flow of network... remove kernel_size change in the last layer
    def __init__(self, input_channel, output_channel, kernel_size=5):
        super(DecoderLayer, self).__init__()

        self.decoding = nn.Sequential(
                nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride=2, padding=2),
                nn.ReLU(inplace=True)
                )

    def forward(self, x1, x2=None, concat=True):
        x1 = self.decoding(x1)

        if not concat:
            return x1
        else:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                            diffY // 2, diffY - diffY//2))

            x = torch.cat([x2, x1], dim=1)

            return x
