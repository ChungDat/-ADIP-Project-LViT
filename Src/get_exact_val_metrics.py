import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import Config as config
from Load_Dataset import ValGenerator, ImageToImage2D
from nets.LViT import LViT
from utils import WeightedDiceBCE, iou_on_batch, read_text
import numpy as np

def get_val_metrics(task_name, session_name):
    # Set config values dynamically
    config.task_name = task_name
    config.test_session = session_name
    config.test_dataset = './datasets/' + task_name + '/Test_Folder/'
    config.val_dataset = './datasets/' + task_name + '/Val_Folder/'
    
    # Load dataset
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    val_text = read_text(config.val_dataset + 'Val_text.xlsx')
    val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Load model
    model_path = f"./{task_name}/LViT/{session_name}/models/best_model-LViT.pth.tar"
    checkpoint = torch.load(model_path, map_location='cuda')
    
    config_vit = config.get_CTranS_config()
    model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    
    loss_sum = 0
    iou_sum = 0
    dice_sum = 0
    total_samples = 0
    
    with torch.no_grad():
        for sampled_batch, names in val_loader:
            images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            if text.shape[1] > 10:
                text = text[:, :10, :]
            images, masks, text = images.cuda(), masks.cuda(), text.cuda()
            
            preds = model(images, text)
            loss = criterion(preds, masks.float())
            dice = criterion._show_dice(preds, masks.float())
            iou = iou_on_batch(masks, preds)
            
            n = len(images)
            loss_sum += loss.item() * n
            dice_sum += float(dice) * n
            iou_sum += float(iou) * n
            total_samples += n
            
    avg_loss = loss_sum / total_samples
    avg_dice = dice_sum / total_samples
    avg_iou = iou_sum / total_samples
    
    print(f"{task_name} ({session_name}) Exact Val -> Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")
    return avg_loss, avg_iou, avg_dice

if __name__ == '__main__':
    get_val_metrics("Covid19", "Supervised")
    get_val_metrics("Covid19", "Semi_supervised")
    get_val_metrics("MosMedDataPlus", "Supervised")
    get_val_metrics("MosMedDataPlus", "Semi_supervised")
