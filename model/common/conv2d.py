import torch.nn as nn
from torch.nn import functional as F

"""
Wrap torch.nn.Conv2d for the simpler using
"""


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in 'torch.nn.Conv2d'
        :param norm(nn.Module, optional), a normalization layer
        :param activation(callable(Tensor) -> Tensor), a callable activation function

        It assumes that norm layer is used before activation
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x