import torch.nn as nn

class LeakyRELU(nn.LeakyReLU):
    def __init__(self, y_deviation, negative_slope=0.01):
        super(LeakyRELU, self).__init__(negative_slope=negative_slope)
        self.x_deviation = y_deviation/negative_slope
        self.y_deviation = y_deviation

    def forward(self, x):
        return super().forward(x-self.x_deviation)+self.y_deviation
