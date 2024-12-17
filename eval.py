import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from utils.common import compute_confusion_matrix, calculate_metrics

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    NUM_CLASSES = 2
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, total=len(dataloader)):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            preds_seg = torch.argmax(outputs, dim=1)
            masks_seg = torch.argmax(masks, dim=1)

            for pred, mask in zip(preds_seg, masks_seg):
                cm = compute_confusion_matrix(
                    pred.cpu().numpy(),
                    mask.cpu().numpy(),
                    NUM_CLASSES
                )
                confusion_matrix += cm
        
        # TODO: add evaluation metrics here ??
        pixel_accuracy, mean_pixel_accuracy, iou, mean_iou = calculate_metrics(confusion_matrix)
        print('EVAL METRICS: ')
        print(f"Pixel Accuracy: {pixel_accuracy:.4f} | Mean Pixel Accuracy: {mean_pixel_accuracy:.4f} | Mean IoU: {mean_iou:.4f} | IoU per Class: {iou}")
            
    
    return running_loss / len(dataloader)