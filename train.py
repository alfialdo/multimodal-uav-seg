
from omegaconf import OmegaConf
import torch
from torchvision import transforms
import os

from trainer import train_one_epoch
from eval import evaluate
from utils import EarlyStopper
from utils import save_model
from utils.common import get_loss_function, get_optimizer, get_dataloaders, write_config

config = OmegaConf.load('config.yaml')
dataset_cfg = config.dataset
trainer_cfg = config.trainer
ckpt_dir = config.trainer.checkpoint.save_dir

BEST_TRAIN_LOSS = float('inf')
BEST_VAL_LOSS = float('inf')

# Get data loaders
transform = transforms.Compose([
        transforms.Resize(dataset_cfg.image_size),
        transforms.ToTensor()
    ])

train_dataloader, val_dataloader = get_dataloaders(dataset_cfg, transform)


# Define the model
if config.model.name == 'VanillaUNet':
    from model.UNet import VanillaUNet
    model = VanillaUNet(in_channels=3, start_out_channels=64, num_class=1, size=4, padding=1)
elif config.model.name == 'DyUNet':
    from model.DyUNet import DyUNet
    model = DyUNet(in_channels=3, start_out_channels=32, num_class=1, size=4, padding=1)
elif config.model.name == 'ThinDyUNet':
    from model.ThinDyUNet import ThinDyUNet
    model = ThinDyUNet(in_channels=3, start_out_channels=64, num_class=1, size=6, padding=1)
elif config.model.name == 'TransUNet':
    from model.TransUNet import TransUNet
    enc_params = config.model.encoder
    img_size = dataset_cfg.image_size[0]
    size=3
    encoder_cfg = dict(
        patch_size=enc_params.patch_size,
        n_trans=enc_params.n_transformer,
        projection_dim=enc_params.projection_dim,
        mlp_head_units=enc_params.mlp_head_units,
        num_heads=enc_params.num_heads,
    )
    encoder_cfg['num_patches'] = ((img_size//(2**size)) // encoder_cfg['patch_size']) ** 2
    encoder_cfg['feed_forward_dim'] = encoder_cfg['projection_dim'] * 2
    model = TransUNet(in_channels=3, start_out_channels=64, num_class=1, size=3, padding=1, encoder_cfg=encoder_cfg)
elif config.model.name == 'MaskRCNN':
    from model.MaskRCNN import MaskRCNN
    model = MaskRCNN(num_class=1, n_hidden_layer=256).model

criterion = get_loss_function(trainer_cfg.loss_fn)
optimizer = get_optimizer(trainer_cfg.optimizer, model, trainer_cfg.lr)

if trainer_cfg.scheduler.type == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=trainer_cfg.scheduler.factor, patience=trainer_cfg.scheduler.patience, cooldown=trainer_cfg.scheduler.cooldown)

device = torch.device(f'cuda:{trainer_cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
early_stopper = EarlyStopper(
    patience = int(trainer_cfg.early_stop.patience), 
    min_delta = float(trainer_cfg.early_stop.min_delta)
)

model.to(device)

# Training
if __name__ == '__main__':
    # write model configuration
    print(f'Running experiment: {ckpt_dir}')
    write_config(ckpt_dir, config)
    

    for epoch in range(trainer_cfg.epochs):
        print(f'Epoch {epoch + 1}/{trainer_cfg.epochs}')
        
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss}')
        
        val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f'Val Loss: {val_loss}')
        
        lr_scheduler.step(val_loss)

        # Save the model
        if train_loss < BEST_TRAIN_LOSS:
            save_model(
                model, optimizer, lr_scheduler, epoch, train_loss,
                f'{ckpt_dir}/{config.model.name}-best-train.pth'
            )
            BEST_TRAIN_LOSS = train_loss
        
        if val_loss < BEST_VAL_LOSS:
            print('Saving validation best model: ', val_loss)
            save_model(
                model, optimizer, lr_scheduler, epoch, val_loss, 
                f'{ckpt_dir}/{config.model.name}-best-val.pth'
            )
            BEST_VAL_LOSS = val_loss

        if early_stopper.early_stop(val_loss):
            print('Early stopping')
            break
