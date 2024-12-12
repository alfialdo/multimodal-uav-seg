import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from glob import glob
from PIL import Image
import cv2

class UAVSegmentation(Dataset):
    def __init__(self, root, num_classes, transforms=None):
        self.root = root
        self.num_classes = num_classes
        self.transforms = transforms
        self.images, self.masks = self.get_file_path(root)
        print(len(self.images), len(self.masks))
        
        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = Image.open(image).convert('L')
        mask = np.load(mask)
        # mask = np.expand_dims(mask, axis=2)
        # mask = np.permute_dims(mask, (1, 2, 0))

        # print(mask.max(), mask.min())
        # print(mask.shape)
        # print(np.unique(mask))

        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

        mask = self.create_mask(mask)

        if self.transforms:
            image = self.transforms(image)

        target = torch.from_numpy(mask.transpose(2, 0, 1))

        return image, target
    
    def create_mask(self, mask):
        mask = np.array(mask)
        new_mask = np.zeros((mask.shape[0], mask.shape[1], self.num_classes))
        for i in range(self.num_classes):
            
            new_mask[:, :, i] = (mask == i).astype(int)
            # print(i, new_mask[:, :, i].max())
        
        return new_mask
    
    def get_file_path(self, root):
        image_list, mask_list = [], []

        for folder in os.listdir(root):
            path = root + folder + '/'
            image_list += sorted(glob(path + 'infrared/*.jpg'))
            mask_list += sorted(glob(path + 'infrared/*.npy'))
        
        return image_list, mask_list
    
class UAVSegmentation256(Dataset):
    def __init__(self, root, num_classes, transforms=None):
        self.root = root
        self.num_classes = num_classes
        self.transforms = transforms
        self.images, self.masks = self.get_file_path(root)
        print(len(self.images), len(self.masks))
        
        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = Image.open(image)
        mask = np.load(mask)
        # mask = np.expand_dims(mask, axis=2)
        # mask = np.permute_dims(mask, (1, 2, 0))

        # print(mask.max(), mask.min())
        # print(mask.shape)
        # print(np.unique(mask))

        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        mask = self.create_mask(mask)

        if self.transforms:
            image = self.transforms(image)

        target = torch.from_numpy(mask.transpose(2, 0, 1))

        return image, target
    
    def create_mask(self, mask):
        mask = np.array(mask)
        new_mask = np.zeros((mask.shape[0], mask.shape[1], self.num_classes))
        for i in range(self.num_classes):
            
            new_mask[:, :, i] = (mask == i).astype(int)
            # print(i, new_mask[:, :, i].max())
        
        return new_mask
    
    def get_file_path(self, root):
        image_list, mask_list = [], []

        for folder in os.listdir(root):
            path = root + folder + '/'
            image_list += sorted(glob(path + 'infrared/*.jpg'))
            mask_list += sorted(glob(path + 'infrared/*.npy'))
        
        return image_list, mask_list
    
if __name__ == '__main__':
    DATASET_PATH = '/mnt/hdd/dataset/uav_dataset/val/'
    NUM_CLASSES = 2

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = UAVSegmentation(DATASET_PATH, NUM_CLASSES, transforms=transform)
    print('Dataset size:', len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print('Dataloader size:', len(dataloader))
    i=0
    for qq, (images, masks) in enumerate(dataloader):
        i+=1
        print(i)
        print(images.shape, masks.shape)
        break