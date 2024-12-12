import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model.vanilla_unet import VanillaUNet
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.voc_2012 import VOC2012DatasetSemantic as VOC2012Dataset

def test():
    NUM_CLASSES = 21
    BATCH_SIZE = 1
    DATASET_PATH = '/mnt/hdd/dataset/VOCdevkit/VOC2012'
    IMAGE_PATH = 'JPEGImages'
    SEMANTIC_MASK_PATH = 'SegmentationClass'
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    # Define the model
    model = VanillaUNet(in_channels=4, out_channels=NUM_CLASSES)
        
    # Load the model
    model.load_state_dict(torch.load('./checkpoints/best_val.pth')['model_state_dict'])
    model.to(device)
    # model.load_state_dict(torch.load('./checkpoints/best_val.pth'))
    
    # Test the model
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    val_dataset = VOC2012Dataset(DATASET_PATH, IMAGE_PATH, SEMANTIC_MASK_PATH, 'val', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for images, masks in val_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            break
    
    # Visualize the first image, mask, and predicted mask
    image = images[0].permute(1, 2, 0).cpu().numpy()
    mask = masks[0].cpu().numpy()
    predicted_mask = outputs[0].argmax(0).cpu().numpy()
    
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
    plt.savefig('output.png')

if __name__ == '__main__':
    test()