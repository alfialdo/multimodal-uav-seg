import torch
from dataset import UAVSegmDataset
from torch.utils.data import DataLoader
import numpy as np
import os

def get_loss_function(loss_fn):
    if loss_fn == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif loss_fn == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Loss function {loss_fn} not supported')
    

def get_optimizer(optimizer, model, lr):
    if optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr)
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
