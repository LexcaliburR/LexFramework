import torch
from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

# import torchvision.models.resnet


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 norm_layer=None, activate_func=None):
        """
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        in_channel == out_channel
        """
        super(BasicBlock, self).__init__()

        if in_channels != out_channels:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.short_cut = None

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3),
                               stride=stride, dilation=dilation, bias=False)
        # TODO: add support for other normalize method
        self.bn1 = nn.BatchNorm2d(out_channels)
        # TODO: add support for other activate function
        self.activate = nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, (3, 3),
                                     stride=1, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        for layer in [self.short_cut[0], self.conv1, self.conv2]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x_in: torch.Tensor):
        out = self.conv1(x_in)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.short_cut is not None:
            shortcut = self.short_cut(x_in)
        else:
            shortcut = x_in

        out += shortcut
        out = self.activate(out)

        return out


class BottleNeckBlock(nn.Module):
    """
    downsampling on 3x3 conv, different from origin article.
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    reference: torchvision.models.resnet
    """
    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 norm_layer=None, activate_func=None):
        super(BottleNeckBlock, self).__init__()

        if in_channels != out_channels:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.short_cut = None

        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, (1, 1), 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.activate = nn.ReLU(inplace=True)

        self.downsample = None
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, (3, 3), stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.conv3 = torch.nn.Conv2d(in_channels, out_channels, (1, 1), 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        for layer in [self.short_cut[0], self.conv1, self.conv2, self.conv3]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x_in):
        out = self.conv1(x_in)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activate(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.short_cut is not None:
            shortcut = self.short_cut(out)
        else:
            shortcut = x_in

        out += shortcut
        out = self.activate(out)

        return out


class BasicStemBlock(nn.Module):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    """
    def __init__(self, in_channels=3, out_channels=64):
        super(BasicStemBlock, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.max_pool(out)
        return out


class ResNet(nn.Module):
    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        """
        :param stem(): a stem module
        :param stages:
        :param num_classes:
        :param out_features:
        :param freeze_at:
        """
        super(ResNet, self).__init__()


    def forward(self, x):
        pass

    def freeze(self, freeze_at=0):
        pass

    def output_shape(self):
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
