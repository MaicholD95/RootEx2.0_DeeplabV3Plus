# main_fine.py
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import random
from dataset import CustomRGBDataset
from DeepLabV3.model import MultiHeadDeeplabV3Plus
from losses import  FocalLoss, DiceLoss,StructureLoss,CombinedLovaszBCELoss
from utils_fine import visualize_outputs,calculate_iou,calculate_f1,set_seed
import copy
import torch.nn as nn
from sklearn.metrics import jaccard_score
import matplotlib
matplotlib.use("Agg")


def train_model(model, dataloaders, criteria, optimizer, scheduler, num_epochs=25, device='cuda'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    writer = SummaryWriter(log_dir='runs/multihead')
    num_gpus = torch.cuda.device_count()
    model = model.to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)

    iou_scores = {'roots': [], 'tips': [], 'sources': []}
    f1_scores = {'roots': [], 'tips': [], 'sources': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        running_root_loss = 0.0
        running_tip_loss = 0.0
        running_source_loss = 0.0
        total_batch =0
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            iou_roots, iou_tips, iou_sources = 0.0, 0.0, 0.0
            num_batches = 0
            f1_roots, f1_tips, f1_sources = 0.0, 0.0, 0.0

            for batch_idx, (inputs, masks) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                masks = masks.to(device)
                roots_masks = masks[:, 0, :, :].unsqueeze(1)
                tips_masks = masks[:, 1, :, :].unsqueeze(1)
                sources_masks = masks[:, 2, :, :].unsqueeze(1)
                optimizer.zero_grad()
                total_batch +=1
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    dice_loss_roots = criteria['roots']['dice'](outputs['roots'], masks[:, 0, :, :].unsqueeze(1))
                    bce_loss_roots = criteria['roots']['bce'](outputs['roots'], masks[:, 0, :, :].unsqueeze(1))
                    root_loss = (dice_loss_roots*0.3+ bce_loss_roots*0.7) #/ 2
                    #root_loss = criteria['roots'](outputs['roots'], masks[:, 1, :, :].unsqueeze(1))

                    tip_loss = criteria['tips'](outputs['tips'], masks[:, 1, :, :].unsqueeze(1))

                    # Sources: Weighted BCE Loss
                    source_loss = criteria['sources'](outputs['sources'], masks[:, 2, :, :].unsqueeze(1))

                    # Total loss (weighted)
                    total_loss = root_loss+ 5*tip_loss + 10*source_loss
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()
                    
                batch_size = inputs.size(0)
                running_loss += total_loss.item() * batch_size
                running_root_loss += root_loss.item() * batch_size
                running_tip_loss += tip_loss.item() * batch_size
                running_source_loss += source_loss.item() * batch_size

                if phase == 'test':

                    preds_roots = torch.sigmoid(outputs['roots']) > 0.7
                    preds_tips = torch.sigmoid(outputs['tips']) > 0.5
                    preds_sources = torch.sigmoid(outputs['sources']) > 0.5

                    iou_roots += calculate_iou(preds_roots, roots_masks)
                    iou_tips += calculate_iou(preds_tips, tips_masks)
                    iou_sources += calculate_iou(preds_sources, sources_masks)

                    f1_roots += calculate_f1(preds_roots, roots_masks)
                    f1_tips += calculate_f1(preds_tips, tips_masks)
                    f1_sources += calculate_f1(preds_sources, sources_masks)

                    num_batches += 1
                
                
            dataset_size =len(dataloaders[phase].dataset)
            #epoch_loss = running_loss / dataset_size
            epoch_root_loss = running_root_loss / dataset_size
            epoch_tip_loss = running_tip_loss / dataset_size
            epoch_source_loss = running_source_loss / dataset_size

            epoch_loss = epoch_root_loss+ epoch_tip_loss +epoch_source_loss
            writer.add_scalar(f'Loss/{phase}/root', epoch_root_loss, epoch)
            writer.add_scalar(f'Loss/{phase}/tip', epoch_tip_loss, epoch)
            writer.add_scalar(f'Loss/{phase}/source', epoch_source_loss, epoch)
            writer.add_scalar(f'Loss/{phase}/epochLoss', epoch_loss, epoch)

            print('-------------------------------------------')
            print(f'{phase} Loss: {epoch_loss:.4f}')
            print(f'Root Loss: {epoch_root_loss:.4f}')
            print(f'Tip Loss: {epoch_tip_loss:.4f}')
            print(f'Source Loss: {epoch_source_loss:.4f}')
            print('-------------------------------------------')

            
            if phase == 'test':
                if (epoch + 1) % 20 == 0:
                    visualize_outputs(inputs, masks, outputs, num_images=5)
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'epoch_model_.pth')    

                iou_roots /= num_batches
                iou_tips /= num_batches
                iou_sources /= num_batches

                f1_roots /= num_batches
                f1_tips /= num_batches
                f1_sources /= num_batches

                f1_scores['roots'].append(f1_roots)
                f1_scores['tips'].append(f1_tips)
                f1_scores['sources'].append(f1_sources)

                iou_scores['roots'].append(iou_roots)
                iou_scores['tips'].append(iou_tips)
                iou_scores['sources'].append(iou_sources)

                writer.add_scalar(f'IoU/roots', iou_roots, epoch)
                writer.add_scalar(f'IoU/tips', iou_tips, epoch)
                writer.add_scalar(f'IoU/sources', iou_sources, epoch)
                writer.add_scalar(f'F1/roots', f1_roots, epoch)
                writer.add_scalar(f'F1/tips', f1_tips, epoch)
                writer.add_scalar(f'F1/sources', f1_sources, epoch)

                print(f'F1 Score Roots: {f1_roots:.4f}')
                print(f'F1 Score Tips: {f1_tips:.4f}')
                print(f'F1 Score Sources: {f1_sources:.4f}')
                print(f'IoU Roots: {iou_roots:.4f}')
                print(f'IoU Tips: {iou_tips:.4f}')
                print(f'IoU Sources: {iou_sources:.4f}')
                print(f'epoch loss : {epoch_loss} - best loss : {best_loss}')
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'best_model.pth')

                    print('Best model updated.')
                    print(f'Best F1 Score Roots: {f1_roots:.4f}')
                    print(f'Best F1 Score Tips: {f1_tips:.4f}')
                    print(f'Best F1 Score Sources: {f1_sources:.4f}')
                    print(f'Best IoU Roots: {iou_roots:.4f}')
                    print(f'Best IoU Tips: {iou_tips:.4f}')
                    print(f'Best IoU Sources: {iou_sources:.4f}')

                scheduler.step(epoch_loss)

        print()

    print(f'Training complete. Best val loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    writer.close()
    return model

from glob import glob
if __name__ == "__main__":

    
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # Set random seeds
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Percorsi del dataset
    dataset_path = r"D:\WORK\DATA\root\semantic_segm_deeplabv3+_multiclass_masks\Cropped_512_tip_aug_mixed_Res"
    train_json_files = glob(dataset_path + '\\train\\*.json')
    test_json_files = glob(dataset_path + '\\test\\*.json')
    #test_image_dir = dataset_path + '\\test\\'

    train_image_dir = dataset_path + '\\train\\'
    test_image_dir = dataset_path + '\\test\\'
    train_dataset = CustomRGBDataset(json_files=train_json_files, image_dir=train_image_dir, phase='train',isTraining=True)
    test_dataset = CustomRGBDataset(json_files=test_json_files, image_dir=test_image_dir, phase='test',isTraining=True)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True),
        'test': DataLoader(test_dataset, batch_size=6, shuffle=False, drop_last=True)
    }


    # train_dataset = CustomRGBDataset(
    #     json_files=train_json_files,
    #     image_dir=train_image_dir,
    #     transform='train',
    #     isTraining=True
    # )
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # positive_pixels_tips = 0
    # negative_pixels_tips = 0

    # for images, masks in train_loader:
    #     # Le maschere hanno la forma [batch_size, num_classes, height, width]
    #     tips_mask = masks[:, 2, :, :]  # Indice 1 per la classe 'tips' , 2 source

    #     positive_pixels_tips += torch.sum(tips_mask == 1).item()
    #     negative_pixels_tips += torch.sum(tips_mask == 0).item()

    # pos_weight_tips = negative_pixels_tips / positive_pixels_tips
    # print(f"pos_weight_tips: {pos_weight_tips}")

    #pos_weight_tips = 524
    #pos_weight_source = 3069


    def visualize_mask(mask, img,idx):
        import numpy as np

        # Unnormalize the image
        mean = np.array([0.485, 0.456, 0.406])  # Mean used in normalization
        std = np.array([0.229, 0.224, 0.225])   # Std used in normalization

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        channels = ['Root Mask', 'Tip Mask', 'Source Mask']

        if img.shape[0] == 3:
            img = img.permute(1, 2, 0).numpy()  # Transpose and convert to numpy
        else:
            img = img.numpy()  # If it's already in [H, W, C]

        # Unnormalize the image
        img = (img * std) + mean
        img = np.clip(img, 0, 1)  # Ensure the pixel values are in [0, 1]

        # Display the original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Display the masks
        for i in range(3):
            axes[i + 1].imshow(mask[i, :, :], cmap='gray')
            axes[i + 1].set_title(f'{channels[i]} Mask')
            axes[i + 1].axis('off')

        plt.tight_layout()
        # Save the figure instead of showing it
        plt.savefig(r'D:\WORK\DATA\root\semantic_segm_multi_head_3class\mask_ex\\'+f'visualization{idx}.png')
        plt.close(fig)


    # from losses import TverskyLoss

    # #Load a batch of 5 masks from the dataset
    # for idx in range(30):
    #     import random 
    #     number =  random.randint(0, 1000)

    #     image, mask = train_dataset[number]  
    #     print(f"Mask {idx+1}")
    #     visualize_mask(mask.numpy(),image,idx)

    model = MultiHeadDeeplabV3Plus(pretrained_backbone_path=r'D:\WORK\DATA\root\semantic_segm_multi_head_3class\best_deeplabv3_resnet101_voc_os16.pth').to(device)  
    pos_weight_tips = 524
    pos_weight_source = 3069
    criteria = {
        'roots': {'dice': DiceLoss(), 'bce': nn.BCEWithLogitsLoss()},  # Dice + BCE Loss for precise root segmentation
        # 'tips': TverskyLoss(alpha=0.6, beta=0.4),
        # 'sources': TverskyLoss(alpha=0.7, beta=0.3)
        'tips':  nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20)),#FocalLoss(alpha=3, gamma=2), 
        'sources': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(60))  # Weighted BCE Loss for source localization
    }
    
    # optimizers = {
    #     'shared': optim.Adam(model.get_shared_parameters(), lr=1e-4, weight_decay=1e-5),
    #     'roots': optim.Adam(model.get_root_parameters(), lr=1e-4, weight_decay=1e-5),
    #     'tips': optim.Adam(model.get_tip_parameters(), lr=1e-4, weight_decay=1e-5),
    #     'sources': optim.Adam(model.get_source_parameters(), lr=1e-4, weight_decay=1e-5)
    # }
    lr_shared = 1e-4
    lr_roots = 1e-4
    lr_tips = 1e-4
    lr_sources = 1e-4

    optimizer = optim.Adam([
        {'params': model.get_shared_parameters(), 'lr': lr_shared},
        {'params': model.get_root_parameters(), 'lr': lr_roots},
        {'params': model.get_tip_parameters(), 'lr': lr_tips},
        {'params': model.get_source_parameters(), 'lr': lr_sources}
    ])
    #scheduler = lr_scheduler.StepLR(optimizers['shared'], step_size=25, gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=15)
    #scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=0.1)
    model = train_model(model, dataloaders, criteria, optimizer, scheduler, num_epochs=200, device=device)






