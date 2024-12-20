import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop
from torchinfo import summary

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CNNBlock, self).__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cnn_block(x)
        return x


class MultiCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, n_conv):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(n_conv):
            self.layers.append(CNNBlock(in_channels, out_channels, padding=padding))
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, size=4):
        super().__init__()
        self.encoder_layers = nn.ModuleList()

        for _ in range(size):
            self.encoder_layers.append(MultiCNNBlock(in_channels, out_channels, padding, n_conv=2))
            self.encoder_layers.append(nn.MaxPool2d(2,2))
            in_channels = out_channels
            out_channels *= 2

        self.encoder_layers.append(MultiCNNBlock(in_channels, out_channels, padding, n_conv=2))

    def forward(self, x):
        route_connection = []

        for layer in self.encoder_layers:
            if isinstance(layer, MultiCNNBlock):
                x = layer(x)
                route_connection.append(x)
                # print('Conv', x.size())
            else:
                x = layer(x)
                # print('Down sample', x.size())

        return x, route_connection
    

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, padding, size=4):
        super().__init__()
        self.decoder_layers = nn.ModuleList()
        self.padding = padding

        for _ in range(size):
            # Use conv transpose for upsampling
            self.decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.decoder_layers.append(MultiCNNBlock(in_channels, out_channels, padding, n_conv=2))
            in_channels //= 2
            out_channels //= 2

        # Use normal conv to remove bn and activation function
        self.decoder_layers.append(nn.Conv2d(in_channels, num_class, kernel_size=1))
        

    def forward(self, x, routes_connection):
        routes_connection.pop(-1)
        for layer in self.decoder_layers:
            if isinstance(layer, MultiCNNBlock):
                # match spatial size by using center crop
                if self.padding == 0:
                    routes_connection[-1] = center_crop(routes_connection[-1], x.shape[-1])

                # concat channels
                x = torch.cat([x, routes_connection.pop(-1)], dim=1)
                x = layer(x)
                # print('Dec Conv', x.size())
            else:
                x = layer(x)
                # print('Up sample', x.size())

        return x
    

class DyUNet(nn.Module):
    def __init__(self,in_channels, start_out_channels, num_class, size, padding=0):
        super().__init__()
        self.encoder = Encoder(in_channels, start_out_channels, padding=padding, size=size)
        self.decoder = Decoder(
            start_out_channels*(2**size), start_out_channels*(2**(size-1)),
            num_class, padding=padding, size=size
        )

    def forward(self, x):
        enc_out, routes = self.encoder(x)
        out = self.decoder(enc_out, routes)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 3, 240, 240)
    model = DyUNet(
        in_channels=3,
        start_out_channels=32,
        num_class=1,
        size=4,
        padding=1,
    )
    print(summary(model, input_data=x, col_width=20, depth=5, row_settings=["depth", "var_names"], col_names=["input_size", "kernel_size", "output_size", "params_percent"]))

