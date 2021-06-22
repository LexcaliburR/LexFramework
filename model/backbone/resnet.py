import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        pass

    def forward(self, x):
        x = x + 10
        print(x)
        pass


class BottelNeckBlock(nn.Module):
    def __init__(self):
        super(BottelNeckBlock, self).__init__()
        pass

    def forward(self, x):
        pass


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        pass

    def forward(self, x):
        pass

if __name__ == '__main__':
    a = BasicBlock()
    a(2)