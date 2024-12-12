import os
# import argparse
import numpy as np
import PIL.Image as Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.voc_2012 import VOC2012DatasetSemantic as VOC2012Dataset
from model.vanilla_unet import VanillaUNet
from trainer import train_one_epoch
from eval import evaluate
from utils import EarlyStopper
from utils import saveModel

from dvclive import Live
import dvc.api

if __name__ == '__main__':
    params = dvc.api.params_show()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--gpuid', type=int, help='GPU ID to use', default=0)
    # args = parser.parse_args()

    DATASET_PATH = '/mnt/hdd/dataset/VOCdevkit/VOC2012'
    IMAGE_PATH = 'JPEGImages'
    SEMANTIC_MASK_PATH = 'SegmentationClass'

    BATCH_SIZE = params['train']['batch_size']
    NUM_EPOCHS = params['train']['epochs']
    LEARNING_RATE = params['train']['lr']
    GPU_ID = params['train']['gpus']

    NUM_CLASSES = params['data']['num_classes']
    NUM_WORKER = params['data']['num_workers']

    SCH_FACTOR = params['scheduler']['factor']
    SCH_PATIENCE = params['scheduler']['patience']
    SCH_COOLDOWN = params['scheduler']['cooldown']

    ES_PATIENCE = params['early_stop']['patience']
    ES_MIN_DELTA = params['early_stop']['min_delta']
    ES_MODE = params['early_stop']['mode']

    BEST_TRAIN_LOSS = float('inf')
    BEST_VAL_LOSS = float('inf')

    SEL_CRITERION = params['setup']['criterion']
    SEL_OPTIMIZER = params['setup']['optimizer']
    SEL_SCHEDULER = params['setup']['scheduler']

    # Define the transformation
    # Resize the image to 256x256 and convert it to a tensor
    # The mask is also resized to 256x256
    # ToTensor() converts the image to a tensor and normalizes the pixel values to [0, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load the dataset
    train_dataset = VOC2012Dataset(DATASET_PATH, IMAGE_PATH, SEMANTIC_MASK_PATH, 'train', transform=transform)
    val_dataset = VOC2012Dataset(DATASET_PATH, IMAGE_PATH, SEMANTIC_MASK_PATH, 'val', transform=transform)

    print('Train dataset size:', len(train_dataset))
    print('Val dataset size:', len(val_dataset))

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)

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

    with Live(save_dvc_exp=True) as live:
        live.log_param("epochs", NUM_EPOCHS)
        live.log_param("batch_size", BATCH_SIZE)
        live.log_param("num_classes", NUM_CLASSES)
        live.log_param("gpu_id", GPU_ID)
        live.log_param("learning_rate", LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
            train_loss, train_dice_loss, train_dice_metrics = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
            print(f'Train Loss: {train_loss} | Dice loss: {train_dice_loss} | Dice metrics: {train_dice_metrics}')
            val_loss, val_dice_loss, val_dice_metrics = evaluate(model, val_dataloader, criterion, device)
            print(f'Val Loss: {val_loss} | Dice loss: {val_dice_loss} | Dice metrics: {val_dice_metrics}')
            
            lr_scheduler.step(val_loss)

            # Save the model
            if train_loss < BEST_TRAIN_LOSS:
                saveModel(model, optimizer, lr_scheduler, epoch, train_loss, 'checkpoints/best_train.pth')
            if val_loss < BEST_VAL_LOSS:
                saveModel(model, optimizer, lr_scheduler, epoch, val_loss, 'checkpoints/best_val.pth')
            
            live.log_metric("train/loss", train_loss)
            live.log_metric("val/loss", val_loss)
            live.next_step()

            if early_stopper.early_stop(val_loss):
                print('Early stopping')
                break

