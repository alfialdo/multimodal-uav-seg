import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import CNNBlock


class CNNBackbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cnn_blocks = nn.ModuleList()

        for _ in range(3):
            self.cnn_blocks.append(CNNBlock(in_channels, out_channels, kernel_size=3, padding=1))
            self.cnn_blocks.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
            out_channels *= 2

    def forward(self, x):
        route_connection = []

        for layer in self.cnn_blocks:
            if isinstance(layer, CNNBlock):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
        
        return x, route_connection
    

    

class TransUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()

        

    def forward(self, x):

        

        return x
    

if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    model = CNNBackbone(3, 64)
    x, routes = model(x)
    print(routes[0].size(), routes[1].size(), routes[2].size(), x.size())