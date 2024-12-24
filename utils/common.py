import torch
from dataset import UAVSegmDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import segmentation_models_pytorch as smp

def get_loss_function(loss_fn):
    if loss_fn == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif loss_fn == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()
    elif loss_fn == 'DiceLoss':
        return smp.losses.DiceLoss(mode='binary')
    elif loss_fn == 'IoULoss':
        return smp.losses.JaccardLoss(mode='binary')
    else:
        raise ValueError(f'Loss function {loss_fn} not supported')
    

def get_optimizer(optimizer, model, lr):
    if optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    else:
        raise ValueError(f'Optimizer {optimizer} not supported')
                         

def get_dataloaders(config, tsfm, test=False):
    if test:
        test_dataset = UAVSegmDataset(
            os.path.join(config.root, 'test'),
            os.path.join(config.root_mask, 'test'),
            transforms=tsfm,
            num_sequences=config.test_sequences
        )
        print('Test dataset size:', len(test_dataset))

        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        return test_dataloader

    train_dataset = UAVSegmDataset(
        os.path.join(config.root, 'train'),
        os.path.join(config.root_mask, 'train'),
        transforms=tsfm,
        num_sequences=config.train_sequences
    )
    print('Train dataset size:', len(train_dataset))

    val_dataset = UAVSegmDataset(
        os.path.join(config.root, 'val'),
        os.path.join(config.root_mask, 'val'),
        transforms=tsfm,
        num_sequences=config.val_sequences
    )
    print('Val dataset size:', len(val_dataset))

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Iterate over the train dataloader
    for images, masks in train_dataloader:
        print('Train batch size:', images.size())
        break

    # Iterate over the val dataloader
    for images, masks in val_dataloader:
        print('Val batch size:', images.size())
        break

    # for idx, (image, mask) in enumerate(train_dataset):
    #     print('Image shape:', image.shape)
    #     print('Mask shape:', mask.shape)
    #     break

    return train_dataloader, val_dataloader


def pixel_accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum().item()
    total = true_mask.numel()
    return correct / total


def seg_miou(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    intersection = (pred_mask & true_mask).float().sum((1, 2, 3))
    union = (pred_mask | true_mask).float().sum((1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def dice_coeff(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    intersection = (pred_mask & true_mask).float().sum((1, 2, 3))
    dice = (2. * intersection + 1e-6) / (pred_mask.float().sum((1, 2, 3)) + true_mask.float().sum((1, 2, 3)) + 1e-6)
    return dice.mean().item()