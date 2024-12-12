import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model.vanilla_unet import VanillaUNetDoubleConv as VanillaUNet
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2

from dataset.uav_segmentation import UAVSegmentation

def compute_confusion_matrix(pred, target, num_classes):
    # print(pred.shape, target.shape)
    pred = pred.flatten()
    target = target.flatten()
    mask = (target >= 0) & (target < num_classes)
    # print(pred.shape, mask.shape, target.shape)
    return np.bincount(
        num_classes * target[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)

def calculate_metrics(confusion_matrix):
    tp = np.diag(confusion_matrix)
    sum_rows = confusion_matrix.sum(axis=1)
    sum_cols = confusion_matrix.sum(axis=0)
    total_pixels = confusion_matrix.sum()

    pixel_accuracy = tp.sum() / total_pixels
    mean_pixel_accuracy = np.mean(tp / np.maximum(sum_rows, 1))
    iou = tp / np.maximum(sum_rows + sum_cols - tp, 1)
    mean_iou = np.mean(iou)

    return pixel_accuracy, mean_pixel_accuracy, iou, mean_iou

def test():
    NUM_CLASSES = 2
    BATCH_SIZE = 5
    DATASET_PATH_TEST = '/mnt/hdd/dataset/uav_dataset/val/'

    device = torch.device('cuda:0')
    # Define the model
    model = VanillaUNet(in_channels=3, out_channels=NUM_CLASSES)
        
    # Load the model
    model.load_state_dict(torch.load('./checkpoints/best_train_uav_dconv.pth', map_location='cuda:0')['model_state_dict'])
    model.to(device)
    # model.load_state_dict(torch.load('./checkpoints/best_val.pth'))
    
    # Test the model
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = UAVSegmentation(DATASET_PATH_TEST, NUM_CLASSES, transforms=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    criterion = torch.nn.CrossEntropyLoss()
    
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
                    num_classes=NUM_CLASSES
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
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean Pixel Accuracy: {mean_pixel_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"IoU per Class: {iou}")
    print(f"Test Loss: {running_loss / len(test_dataloader)}")
    
    

if __name__ == '__main__':
    test()