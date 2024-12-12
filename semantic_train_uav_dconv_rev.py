import os
# import argparse
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

from dataset.uav_segmentation import UAVSegmentation256 as UAVSegmentation
from model.vanilla_unet import VanillaUNetDoubleConv as VanillaUNet
from trainer import train_one_epoch
from eval import evaluate
from utils import EarlyStopper
from utils import saveModel

DATASET_PATH_TRAIN = '/mnt/hdd/dataset/uav_dataset/train/'
DATASET_PATH_VAL = '/mnt/hdd/dataset/uav_dataset/test/'
DATASET_PATH_TEST = '/mnt/hdd/dataset/uav_dataset/val/'

BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
GPU_ID = 0

NUM_CLASSES = 2
NUM_WORKER = 30

SCH_FACTOR = 0.15
SCH_PATIENCE = 15
SCH_COOLDOWN = 5

ES_PATIENCE = 30
ES_MIN_DELTA = 0.001
ES_MODE = "min"

BEST_TRAIN_LOSS = float('inf')
BEST_VAL_LOSS = float('inf')

SEL_CRITERION = "CrossEntropyLoss"
SEL_OPTIMIZER = "AdamW"
SEL_SCHEDULER = "ReduceLROnPlateau"

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

train_dataset = UAVSegmentation(DATASET_PATH_TRAIN, NUM_CLASSES, transforms=transform)
val_dataset = UAVSegmentation(DATASET_PATH_VAL, NUM_CLASSES, transforms=transform)
test_dataset = UAVSegmentation(DATASET_PATH_TEST, NUM_CLASSES, transforms=transform)


print('Train dataset size:', len(train_dataset))
print('Val dataset size:', len(val_dataset))
print('Test dataset size:', len(test_dataset))

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)

# Iterate over the train dataloader
for images, masks in train_dataloader:
    print('Train batch size:', images.size())
    break

# Iterate over the val dataloader
for images, masks in val_dataloader:
    print('Val batch size:', images.size())
    break

for idx, (image, mask) in enumerate(train_dataset):
    print('Image shape:', image.shape)
    print('Mask shape:', mask.shape)
    break

# Define the model
model = VanillaUNet(in_channels=3, out_channels=NUM_CLASSES)

if SEL_CRITERION == 'CrossEntropyLoss':
    criterion = torch.nn.CrossEntropyLoss()
elif SEL_CRITERION == 'BCEWithLogitsLoss':
    criterion = torch.nn.BCEWithLogitsLoss()

if SEL_OPTIMIZER == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif SEL_OPTIMIZER == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
elif SEL_OPTIMIZER == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

if SEL_SCHEDULER == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=SCH_FACTOR, patience=SCH_PATIENCE, cooldown=SCH_COOLDOWN)

device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
early_stopper = EarlyStopper(patience = int(ES_PATIENCE), 
                            min_delta = float(ES_MIN_DELTA))

model.to(device)

# Training
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    train_loss, train_dice_loss, train_dice_metrics = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
    print(f'Train Loss: {train_loss} | Dice loss: {train_dice_loss} | Dice metrics: {train_dice_metrics}')
    val_loss, val_dice_loss, val_dice_metrics = evaluate(model, val_dataloader, criterion, device)
    print(f'Val Loss: {val_loss} | Dice loss: {val_dice_loss} | Dice metrics: {val_dice_metrics}')
    
    lr_scheduler.step(val_loss)

    # Save the model
    if train_loss < BEST_TRAIN_LOSS:
        saveModel(model, optimizer, lr_scheduler, epoch, train_loss, 'checkpoints/best_train_uav_dconv_rev.pth')
    if val_loss < BEST_VAL_LOSS:
        saveModel(model, optimizer, lr_scheduler, epoch, val_loss, 'checkpoints/best_val_uav_dconv_rev.pth')

    if early_stopper.early_stop(val_loss):
        print('Early stopping')
        break
