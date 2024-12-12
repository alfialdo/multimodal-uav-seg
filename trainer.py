import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils.dice_loss import dice_loss, dice_coeff, multiclass_dice_coeff

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(dataloader, total=len(dataloader)):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        # outputs = torch.argmax(outputs, dim=1).float()
        # print(outputs.dtype, masks.dtype)
        loss = criterion(outputs, masks)
        loss_dice = criterion(outputs, masks)
        # loss_dice = dice_loss(
        #                     F.softmax(outputs, dim=1).float(),
        #                     F.one_hot(masks, 21).permute(0, 3, 1, 2).float(),
        #                     multiclass=True
        #                 )
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        dice_metrics = multiclass_dice_coeff(outputs[:, 1:], masks[:, 1:], reduce_batch_first=False)
    
    return running_loss / len(dataloader), loss_dice.item(), dice_metrics.mean().item()