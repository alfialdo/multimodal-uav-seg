import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(dataloader, total=len(dataloader)):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)

        loss = criterion(outputs, masks)        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # dice_metrics = multiclass_dice_coeff(outputs[:, 1:], masks[:, 1:], reduce_batch_first=False)
    
    return running_loss / len(dataloader) #, dice_metrics.mean().item()