import torch
from torch import nn
# import torchvision.models.resnet


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride_changed=False, dilation=1, norm_layer=None, activate_func=None):
        """
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        in_channel == out_channel
        """
        super(BasicBlock, self).__init__()

        if stride_changed:
            self.conv1 = torch.nn.Conv2d(in_channel, out_channel, (3, 3), stride=2, dilation=dilation)
        else:
            self.conv1 = torch.nn.Conv2d(in_channel, out_channel, (3, 3), stride=1, dilation=dilation)

        # TODO: add support for other normalize method
        self.bn1 = nn.BatchNorm2d(out_channel)
        # TODO: add support for other activate function
        self.activate = nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(in_channel, out_channel, (3, 3), stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x_in: torch.Tensor):
        identity = x_in

        out = self.conv1(x_in)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # Error: on first layer of each group, W/H of input is double than the W/H of output in the block
        out = out + identity
        out = self.activate(out)

        return out


class BottleNeckBlock(nn.Module):
    """
    downsampling on 3x3 conv, different from origin article.
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    reference: torchvision.models.resnet
    """
    def __init__(self, in_channel, out_channel, stride_changed=False):
        super(BottleNeckBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(out_channel, out_channel, (1, 1), 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.activate = nn.ReLU(inplace=True)

        self.downsample = None
        if stride_changed:
            self.conv2 = torch.nn.Conv2d(out_channel, out_channel, (3, 3), 1)
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channel, out_channel * 4, (1, 1), 2),
                nn.BatchNorm2d(out_channel * 4)
            )
        else:
            self.conv2 = torch.nn.Conv2d(out_channel, out_channel, (3, 3), 2)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = torch.nn.Conv2d(out_channel, out_channel * 4, (1, 1), 1)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)

    def forward(self, x_in):
        out = self.conv1(x_in)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activate(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            out = self.downsample(out)
        out = self.activate(out)

        return out


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


# if __name__ == '__main__':