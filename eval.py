import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, total=len(dataloader)):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            
            # TODO: add evaluation metrics here ??
            # dice_metrics = multiclass_dice_coeff(outputs[:, 1:], masks[:, 1:], reduce_batch_first=False)
    
    return running_loss / len(dataloader)