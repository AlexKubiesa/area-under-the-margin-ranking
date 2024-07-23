import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, stride=1):
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("Expected out_channels >= in_channels.")
        if stride not in [1, 2]:
            raise ValueError("Expected stride in [1, 2].")
        self.stride = stride
        self.spatial_padding = self._calculate_spatial_padding(stride)
        self.channel_padding = self._calculate_channel_padding(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = F.relu(self.bn1(self.conv1(F.pad(x, self.spatial_padding))))
        residual = self.conv2(residual)
        x = F.avg_pool2d(x, self.stride)
        x = F.pad(x, self.channel_padding)
        output = F.relu(self.bn2(residual + x))
        return output
    
    @staticmethod
    def _calculate_spatial_padding(stride):
        pad_amount = 3 - stride
        pad_before = pad_amount // 2
        pad_after = pad_amount - pad_before
        return (pad_before, pad_after, pad_before, pad_after, 0, 0, 0, 0)
    
    @staticmethod
    def _calculate_channel_padding(in_channels, out_channels):
        pad_after = out_channels - in_channels
        return (0, 0, 0, 0, 0, pad_after, 0, 0)



class ResNet32(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.substack1 = self._make_substack(16, 16)
        self.substack2 = self._make_substack(16, 32, stride=2)
        self.substack3 = self._make_substack(32, 64, stride=2)
        self.fc = nn.Linear(64, num_classes)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.substack1(x)
        x = self.substack2(x)
        x = self.substack3(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-2, -1)
        x = self.fc(x)
        return x
    
    @staticmethod
    def _make_substack(in_channels, out_channels, *, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(4):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
