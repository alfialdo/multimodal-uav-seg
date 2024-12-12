import os

import torch
import torch.nn as nn

import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class VanillaUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super().__init__()
        self.stage1 = nn.Sequential(
                        ConvBlock(in_channels, 64),
                        ConvBlock(64, 128)
                    ) #256x256
        self.ds_1 = nn.MaxPool2d(kernel_size=2, stride=2) #128x128
        self.stage2 = nn.Sequential(
                        ConvBlock(128, 256),
                        ConvBlock(256, 512)
                    ) #128x128
        self.ds2 = nn.MaxPool2d(kernel_size=2, stride=2) #64x64
        self.stage3 = nn.Sequential(
                        ConvBlock(512, 1024),
                        ConvBlock(1024, 2048)
                    ) #64x64
        self.ds3 = nn.MaxPool2d(kernel_size=2, stride=2) #32x32

        self.middle = ConvBlock(2048, 2048) #32x32

        self.us3 = nn.ConvTranspose2d(2048, 2048, kernel_size=2, stride=2) #64x64
        self.stage3_up = nn.Sequential(
                            ConvBlock(2048 + 2048, 1024),
                            ConvBlock(1024, 512)
                        ) #64x64
        self.us2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2) #128x128
        self.stage2_up = nn.Sequential(
                            ConvBlock(512 + 512, 256),
                            ConvBlock(256, 128)
                        ) #128x128
        self.us1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2) #256x256
        self.stage1_up = nn.Sequential(
                            ConvBlock(128 + 128, 64),
                            ConvBlock(64, 32)
                        ) #256x256
        
        self.segmentation_head = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        x_s1 = self.stage1(x)
        x_ds1 = self.ds_1(x_s1)
        x_s2 = self.stage2(x_ds1)
        x_ds2 = self.ds2(x_s2)
        x_s3 = self.stage3(x_ds2)
        x_ds3 = self.ds3(x_s3)

        x_m = self.middle(x_ds3)

        x_us3 = self.us3(x_m)
        x_us3 = torch.cat([x_us3, x_s3], dim=1)
        x_s3u = self.stage3_up(x_us3)

        x_us2 = self.us2(x_s3u)
        x_us2 = torch.cat([x_us2, x_s2], dim=1)
        x_s2u = self.stage2_up(x_us2)

        x_us1 = self.us1(x_s2u)
        x_us1 = torch.cat([x_us1, x_s1], dim=1)
        x_s1u = self.stage1_up(x_us1)
        # x_s1u = torch.cat([x_s3u, x], dim=1)
        # x = self.encoder(x)
        # x = self.middle(x)
        # x = self.decoder(x)
        
        segmentation_output = self.segmentation_head(x_s1u)
        
        return segmentation_output
    
class VanillaUNetDoubleConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super().__init__()
        self.stage1 = nn.Sequential(
                        DoubleConv(in_channels, 64),
                        DoubleConv(64, 128)
                    ) #256x256
        self.ds_1 = nn.MaxPool2d(kernel_size=2, stride=2) #128x128
        self.stage2 = nn.Sequential(
                        DoubleConv(128, 256),
                        DoubleConv(256, 512)
                    ) #128x128
        self.ds2 = nn.MaxPool2d(kernel_size=2, stride=2) #64x64
        self.stage3 = nn.Sequential(
                        DoubleConv(512, 1024),
                        DoubleConv(1024, 2048)
                    ) #64x64
        self.ds3 = nn.MaxPool2d(kernel_size=2, stride=2) #32x32

        self.middle = DoubleConv(2048, 2048) #32x32

        self.us3 = nn.ConvTranspose2d(2048, 2048, kernel_size=2, stride=2) #64x64
        self.stage3_up = nn.Sequential(
                            DoubleConv(2048 + 2048, 1024),
                            DoubleConv(1024, 512)
                        ) #64x64
        self.us2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2) #128x128
        self.stage2_up = nn.Sequential(
                            DoubleConv(512 + 512, 256),
                            DoubleConv(256, 128)
                        ) #128x128
        self.us1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2) #256x256
        self.stage1_up = nn.Sequential(
                            DoubleConv(128 + 128, 64),
                            DoubleConv(64, 32)
                        ) #256x256
        
        self.segmentation_head = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        x_s1 = self.stage1(x)
        x_ds1 = self.ds_1(x_s1)
        x_s2 = self.stage2(x_ds1)
        x_ds2 = self.ds2(x_s2)
        x_s3 = self.stage3(x_ds2)
        x_ds3 = self.ds3(x_s3)

        x_m = self.middle(x_ds3)

        x_us3 = self.us3(x_m)
        x_us3 = torch.cat([x_us3, x_s3], dim=1)
        x_s3u = self.stage3_up(x_us3)

        x_us2 = self.us2(x_s3u)
        x_us2 = torch.cat([x_us2, x_s2], dim=1)
        x_s2u = self.stage2_up(x_us2)

        x_us1 = self.us1(x_s2u)
        x_us1 = torch.cat([x_us1, x_s1], dim=1)
        x_s1u = self.stage1_up(x_us1)
        # x_s1u = torch.cat([x_s3u, x], dim=1)
        # x = self.encoder(x)
        # x = self.middle(x)
        # x = self.decoder(x)
        
        segmentation_output = self.segmentation_head(x_s1u)
        
        return segmentation_output
    
if __name__ == '__main__':
    model = VanillaUNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.size())