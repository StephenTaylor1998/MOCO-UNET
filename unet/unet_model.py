""" Full assembly of the parts to form the complete network """

# from .unet_parts import *
from unet.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class MOCOUNet(UNet):
    def __init__(self, num_classes, unet_channels=3, unet_classes=2, bilinear=False):
        super(MOCOUNet, self).__init__(unet_channels, unet_classes, bilinear)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out = self.pool(x5)
        out = torch.reshape(out, out.shape[:-2])
        # print(out.shape)
        return self.classifier(out)

    def load_weight(self, unet: UNet):
        unet.down1.load_state_dict(self.down1.state_dict(), strict=False)
        unet.down2.load_state_dict(self.down2.state_dict(), strict=False)
        unet.down3.load_state_dict(self.down3.state_dict(), strict=False)
        unet.down4.load_state_dict(self.down4.state_dict(), strict=False)
        return unet


if __name__ == '__main__':
    unet = UNet(n_channels=3, n_classes=2)
    mnet = MOCOUNet(num_classes=128, unet_channels=3, unet_classes=2)

    print('[INFO] origin unet param: \n',
          unet.down1.state_dict()['maxpool_conv.1.double_conv.0.weight'][0, 0])
    unet = mnet.load_weight(unet)
    print('[INFO] moco-unet param: \n',
          unet.down1.state_dict()['maxpool_conv.1.double_conv.0.weight'][0, 0])
    import torch
    out = mnet(torch.randn(2, 3, 224, 224))
