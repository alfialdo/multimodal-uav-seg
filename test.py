import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T

from utils.common import pixel_accuracy, seg_miou, dice_coeff, get_dataloaders
from argparse import ArgumentParser
from omegaconf import OmegaConf

ap = ArgumentParser()
ap.add_argument('--model-path', default='checkpoints/best-val.pth')
args = ap.parse_args()

config = OmegaConf.load('config.yaml')
device = torch.device(f'cuda:{config.trainer.gpu_id}' if torch.cuda.is_available() else 'cpu')

tsfm  = T.Compose([
    T.Resize(config.dataset.image_size),
    T.ToTensor()
])

test_loader = get_dataloaders(config.dataset, tsfm, test=True)

if config.model.name == 'VanillaUNet':
    from model.UNet import VanillaUNet
    model = VanillaUNet(in_channels=3, start_out_channels=32, num_class=1, size=4, padding=1)

model_path = args.model_path
model.load_state_dict(torch.load(
    args.model_path,
    weights_only=True,
)['model_state_dict'])
model.to(device)
model.eval()

if __name__ == '__main__':

    total_acc = 0.0
    total_miou = 0.0
    total_dice = 0.0
    total_batches = len(test_loader)
    
    with torch.no_grad():
        for images, true_masks in tqdm(test_loader, desc='Model Evaluation'):
            images = images.to(device)
            true_masks = true_masks.to(device)            
            outputs = model(images)

            # Decode the predictions logits
            pred_masks = outputs.sigmoid()
            pred_masks = (pred_masks > 0.9).float()

            total_acc += pixel_accuracy(pred_masks, true_masks)
            total_miou += seg_miou(pred_masks, true_masks)
            total_dice += dice_coeff(pred_masks, true_masks)
        
        avg_acc = total_acc / total_batches
        avg_miou = total_miou / total_batches
        avg_dice = total_dice / total_batches
        
        print('TEST METRICS: ')
        print(f"Pixel Accuracy: {avg_acc:.4f} | Mean IoU: {avg_miou:.4f} | Dice Coeff: {avg_dice:.4f}")
            
