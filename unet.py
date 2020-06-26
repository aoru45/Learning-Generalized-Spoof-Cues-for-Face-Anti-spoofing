import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
class DoubleConv(nn.Module):
    """(convolution =&gt; [BN] =&gt; ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class ModifiedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ModifiedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16) # 48
        self.down1 = Down(16, 32) # 24
        self.down2 = Down(32, 64) # 12
        self.down3 = Down(64, 128) # 6
        factor = 2 if bilinear else 1 
        self.down4 = Down(128, 256 // factor) # 3
        self.up1 = Up(256, 128 // factor, bilinear) # 6
        self.up2 = Up(128, 64 // factor, bilinear)# 12 
        self.up3 = Up(64, 32 // factor, bilinear)# 24
        self.up4 = Up(32, 16, bilinear) # 48
        self.regression = OutConv(16, n_classes)# 48
        self.classification = resnet18(pretrained = True)
        in_features = self.classification.fc.in_features
        self.classification.fc = nn.Linear(in_features, 2)
    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)
        d1 = self.up1(e5, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)
        regression = self.regression(d4)
        classification = self.classification(x + regression)
        feat = [e5, d1, d2, d3, d4]
        return regression, classification, feat 
class TripUNet(nn.Module):
    def __init__(self,):
        super(TripUNet, self).__init__()
        self.net = ModifiedUNet(n_channels = 3, n_classes = 3)
    def forward(self,anchor, positive, negative):
        regression_anchor, classification_anchor, feat_anchor = self.net(anchor)
        regression_positive, classification_positive, feat_positive = self.net(positive)
        regression_negative, classification_negative, feat_negative = self.net(negative)
        return  [regression_anchor, regression_positive, regression_negative],\
                [classification_anchor, classification_positive, classification_negative],\
                [feat_anchor, feat_positive, feat_negative]
        
if __name__ == "__main__":
    model = UNet(3, 3)
    x = torch.randn(1,3,64,64)
    print(model(x).size())
