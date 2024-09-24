import torch
from torch import nn

class MultiScaleSemanticConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleSemanticConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        return conv1_out + conv3_out + conv5_out

