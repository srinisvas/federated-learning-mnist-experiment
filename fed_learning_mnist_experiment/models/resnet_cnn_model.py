import torch
import torch.nn as nn
from typing import Optional, Type


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1      = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _make_layer(
    block: Type[BasicBlock],
    in_planes: int,
    planes: int,
    blocks: int,
    stride: int,
) -> nn.Sequential:
    downsample = None
    if stride != 1 or in_planes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = [block(in_planes, planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(block(planes * block.expansion, planes))
    return nn.Sequential(*layers)


class TinyResNet18(nn.Module):
    """
    CIFAR-10 / MNIST friendly ResNet-18 layout (2-2-2-2) with narrow channels.

    Args:
        num_classes: Number of output classes (default 10).
        base_width:  Controls channel widths and total parameter count.
                       8  -> ~0.27 M params
                       12 -> ~0.60 M params
                       16 -> ~1.05 M params
        in_channels: Number of input image channels.
                       3 for RGB (CIFAR-10), 1 for grayscale (MNIST).
    """

    def __init__(
        self,
        num_classes: int = 10,
        base_width: int = 8,
        in_channels: int = 1,          # <-- new: default changed to 1 for MNIST
    ):
        super().__init__()
        self.in_planes = base_width

        # Stem: 3x3, stride 1, no maxpool (suitable for small images like 28x28 and 32x32)
        self.conv1 = nn.Conv2d(
            in_channels, self.in_planes,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1  = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # ResNet-18 stages — spatial downsampling at the start of layers 2-4
        self.layer1 = _make_layer(BasicBlock, base_width,    base_width,   blocks=2, stride=1)
        self.layer2 = _make_layer(BasicBlock, base_width,    base_width*2, blocks=2, stride=2)
        self.layer3 = _make_layer(BasicBlock, base_width*2,  base_width*4, blocks=2, stride=2)
        self.layer4 = _make_layer(BasicBlock, base_width*4,  base_width*8, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(base_width * 8, num_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.momentum = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def tiny_resnet18(
    num_classes: int = 10,
    base_width: int = 8,
    in_channels: int = 1,
) -> TinyResNet18:
    return TinyResNet18(num_classes=num_classes, base_width=base_width, in_channels=in_channels)


if __name__ == "__main__":
    # Sanity checks for both MNIST (1-channel 28x28) and CIFAR (3-channel 32x32)
    for name, c, h in [("MNIST", 1, 28), ("CIFAR-10", 3, 32)]:
        model = tiny_resnet18(in_channels=c)
        x = torch.randn(2, c, h, h)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:10s} | input {x.shape} | output {y.shape} | params {params:,}")