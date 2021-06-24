import torch
from torch import nn
import torchvision.models.resnet

# kernel_size: _size_2_t,
# stride: _size_2_t = 1,
# padding: _size_2_t = 0,
# dilation: _size_2_t = 1,
# groups: int = 1,
# bias: bool = True,
# padding_mode: str = 'zeros'  # TODO: refine this type


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride_changed=False, dilation=1):
        """
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        """
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, (3, 3), stride=1, dilation=dilation)
        if stride_changed:
            self.conv2 = torch.nn.Conv2d(in_channel, out_channel, (3, 3), stride=2, dilation=dilation)
        else:
            self.conv2 = torch.nn.Conv2d(in_channel, out_channel, (3, 3), stride=1, dilation=dilation)

    def forward(self, x_in: torch.Tensor):
        x = self.conv1(x_in)
        x = self.conv2(x)
        return x + x_in


class BottelNeckBlock(nn.Module):
    """
    downsampling on 3x3 conv, different from origin article.
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    reference: torchvision.models.resnet
    """
    def __init__(self, in_channel, out_channel, stride_changed=False):
        super(BottelNeckBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(out_channel, out_channel, (1, 1), 1)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, (3, 3), 1)
        if stride_changed:
            self.conv3 = torch.nn.Conv2d(out_channel, out_channel * 4, (1, 1), 1)
        else:
            self.conv3 = torch.nn.Conv2d(out_channel, out_channel * 4, (1, 1), 2)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + x_in


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        pass

    def forward(self, x):
        pass

def build_resnet(name: str):
    if name == "resnet-18":
        pass
    elif name == "resnet-34":
        pass
    elif name == "resnet-50":
        pass
    elif name == "resnet-101":
        pass
    elif name == "resnet-152":
        pass
    else:
        raise Exception("resnet net model {} is not supported".format(name))


if __name__ == '__main__':
    a = BasicBlock()
    a(2)