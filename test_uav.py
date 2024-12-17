import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from omegaconf import OmegaConf
import numpy as np
import cv2

from model.UNet import VanillaUNetDoubleConv as VanillaUNet
from utils.common import compute_confusion_matrix, calculate_metrics, get_dataloaders, get_loss_function

def test():
    config = OmegaConf.load('config.yaml')
    trainer_cfg = config.trainer
    dataset_cfg = config.dataset
    device = torch.device(f'cuda:{trainer_cfg.gpu_id}')

        
    # Load the model
    model = VanillaUNet(in_channels=3, out_channels=dataset_cfg.num_classes)
    model.load_state_dict(torch.load(trainer_cfg.checkpoint.test_path, map_location=f'cuda:{trainer_cfg.gpu_id}')['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Test the model
    transform = transforms.Compose([
        transforms.Resize(dataset_cfg.image_size),
        transforms.ToTensor()
    ])

    test_dataloader = get_dataloaders(dataset_cfg, transform, test=True)

    criterion = get_loss_function(trainer_cfg.loss_fn)
    confusion_matrix = np.zeros((dataset_cfg.num_classes, dataset_cfg.num_classes), dtype=np.int64)
    image_k = 0
    running_loss = 0.0 
    
    with torch.no_grad():
        for images, masks in test_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)

            loss = criterion(outputs, masks)

            preds_seg = torch.argmax(outputs, dim=1)
            masks_seg = torch.argmax(masks, dim=1)

            for pred, mask in zip(preds_seg, masks_seg):
                cm = compute_confusion_matrix(
                    pred.cpu().numpy(),
                    mask.cpu().numpy(),
                    num_classes=dataset_cfg.num_classes
                )
                confusion_matrix += cm

            running_loss += loss.item()
            
            # Visualize the first image, mask, and predicted mask
            image = images[0].permute(1, 2, 0).cpu().numpy()
            mask = masks[0].argmax(0).cpu().numpy()
            predicted_mask = outputs[0].argmax(0).cpu().numpy()

            cv2.imwrite(f'test_im_dconv/image_{image_k}.png', image*255)
            cv2.imwrite(f'test_im_dconv/mask_{image_k}.png', mask*255)
            cv2.imwrite(f'test_im_dconv/predicted_mask_{image_k}.png', predicted_mask*255)
            
            plt.figure(figsize=(12, 12))
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.axis('off')
            plt.title('Image')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask)
            plt.axis('off')
            plt.title('Mask')
            
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_mask)
            plt.axis('off')
            plt.title('Predicted Mask')
            
            plt.show()
            plt.savefig(f'test_im_dconv/output_{image_k}.png')

            image_k += 1

            break
            
    pixel_accuracy, mean_pixel_accuracy, iou, mean_iou = calculate_metrics(confusion_matrix)
    # TODO: add pixel accuracy to validation step
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean Pixel Accuracy: {mean_pixel_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"IoU per Class: {iou}")
    print(f"Test Loss: {running_loss / len(test_dataloader)}")
    
    

if __name__ == '__main__':
    test()