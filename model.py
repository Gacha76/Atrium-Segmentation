import torch

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, X):
        return self.step(X)
    
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder layers
        self.layer1 = DoubleConv(in_channels=1, out_channels=64)
        self.layer2 = DoubleConv(in_channels=64, out_channels=128)
        self.layer3 = DoubleConv(in_channels=128, out_channels=256)
        self.layer4 = DoubleConv(in_channels=256, out_channels=512)

        # Decoder layers
        self.layer5 = DoubleConv(in_channels=512 + 256, out_channels=256)
        self.layer6 = DoubleConv(in_channels=256 + 128, out_channels=128)
        self.layer7 = DoubleConv(in_channels=128 + 64, out_channels=64)
        self.layer8 = torch.nn.Conv2d(in_channels=64,
                                      out_channels=1,
                                      kernel_size=1)
        
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # Encoder
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)

        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)

        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)

        x4 = self.layer4(x3m)
        
        # Decoder
        x5 = torch.nn.Upsample(scale_factor=2, mode='bilinear')(x4)
        x5 = torch.cat(tensors=[x5, x3], dim=1)
        x5 = self.layer5(x5)

        x6 = torch.nn.Upsample(scale_factor=2, mode='bilinear')(x5)
        x6 = torch.cat(tensors=[x6, x2], dim=1)
        x6 = self.layer6(x6)

        x7 = torch.nn.Upsample(scale_factor=2, mode='bilinear')(x6)
        x7 = torch.cat(tensors=[x7, x1], dim=1)
        x7 = self.layer7(x7)

        # Final Segmentation
        ret = self.layer8(x7)

        return ret
