import torch

def save_model(model, optimizer, lr_scheduler, epoch, loss, path):
    """
    Save the model to the disk

    Args:
    - model: The model to save
    - optimizer: The optimizer used to train the model
    - lr_scheduler: The learning rate scheduler used to train the model
    - epoch: The current epoch
    - loss: The loss at the current epoch

    Returns:    
    - None
    """

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss,
    }, path)