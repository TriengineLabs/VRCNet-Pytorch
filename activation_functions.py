import torch.nn as nn
import torch

class LeakyRELU(nn.LeakyReLU):
    def __init__(self, y_deviation, negative_slope=0.01):
        super(LeakyRELU, self).__init__(negative_slope=negative_slope)
        self.x_deviation = y_deviation/negative_slope
        self.y_deviation = y_deviation

    def forward(self, x):
        return super().forward(x-self.x_deviation)+self.y_deviation

class CustSigmoid(nn.Sigmoid):
    def __init__(self):
        super(CustSigmoid, self).__init__()
        self.scalar = torch.Tensor([6])

    def forward(self, x):
        x = torch.sub(x, 2)
        x = super().forward(x)
        return torch.mul(x, self.scalar)
