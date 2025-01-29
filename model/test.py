
import torch
from torchinfo import summary





if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(x)['out'][0].argmax(0)
    print(out.size())
    # summary(model, input_data=x, col_width=20, depth=5, row_settings=["depth", "var_names"], col_names=["input_size", "kernel_size", "output_size", "params_percent"])


