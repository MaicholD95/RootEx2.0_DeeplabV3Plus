
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import torch
import torch.nn as nn
import copy
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils.utils import visualize_outputs, calculate_iou, calculate_f1
from torch.amp import autocast, GradScaler
import time 
from DeepLabV3.model import MultiHeadDeeplabV3Plus
import torch.optim as optim
from torch.optim import lr_scheduler

def run_training(config, device, dataloaders):
    # resnet50 or resnet101
    model = MultiHeadDeeplabV3Plus(pretrained_backbone_path=config['pretrained_backbone'], backbone='resnet50').to(device)
    # Configure the optimizer
    optimizer = optim.Adam([
        {'params': model.get_shared_parameters(), 'lr': config['optimizer_params']['lr_shared']},
        {'params': model.get_root_parameters(), 'lr': config['optimizer_params']['lr_roots']},
        {'params': model.get_tip_parameters(), 'lr': config['optimizer_params']['lr_tips']},
        {'params': model.get_source_parameters(), 'lr': config['optimizer_params']['lr_sources']}
    ])
    
    # Configure the scheduler
    scheduler_params = config['scheduler_params']
    if scheduler_params['type'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min', 
            factor=scheduler_params['factor'], 
            patience=scheduler_params['patience']
        )
    elif scheduler_params['type'] == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer=optimizer, 
            step_size=scheduler_params['step_size'], 
            gamma=scheduler_params['gamma']
        )
    else:
        scheduler = None

    # Run training
    model, best_metrics = train_model(
        model=model,
        dataloaders=dataloaders,
        criteria=config['criteria'],
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=device,
        experiment_name=config['experiment_name'],
        loss_weights=config['loss_weights'] 
    )
    
    return best_metrics

def train_model(model, dataloaders, criteria, optimizer, scheduler, num_epochs=25, device='cuda', experiment_name='experiment', loss_weights=None):
    if loss_weights is None:
        # Imposta valori di default se non specificati
        loss_weights = {
            'root_dice': 0.3,
            'root_bce': 0.7,
            'tip_loss': 2,
            'source_loss': 3
        }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_epoch = -1
    writer = SummaryWriter(log_dir=f'runs/{experiment_name}')
    num_gpus = torch.cuda.device_count()
    print(f'num_gpus:{num_gpus} ')
    model = model.to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    
    scaler = GradScaler()  # Inizializza lo scaler per il mixed precision

    iou_scores = {'roots': [], 'tips': [], 'sources': []}
    f1_scores = {'roots': [], 'tips': [], 'sources': []}
    # Aggiunta per l'early stopping
    best_f1_sum = 0.0
    epochs_no_improve = 0
    early_stopping_patience = 15
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        running_root_loss = 0.0
        running_tip_loss = 0.0
        running_source_loss = 0.0

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

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
                    
                    dice_loss_roots = criteria['roots']['dice'](outputs['roots'], masks[:, 0, :, :].unsqueeze(1))
                    bce_loss_roots = criteria['roots']['bce'](outputs['roots'], masks[:, 0, :, :].unsqueeze(1))
                    root_loss = (dice_loss_roots * loss_weights['root_dice'] + bce_loss_roots * loss_weights['root_bce'])

                    tip_loss = criteria['tips'](outputs['tips'], masks[:, 1, :, :].unsqueeze(1))
                    source_loss = criteria['sources'](outputs['sources'], masks[:, 2, :, :].unsqueeze(1))

                    # Total loss (weighted)
                    total_loss = root_loss + loss_weights['tip_loss'] * tip_loss + loss_weights['source_loss'] * source_loss
                    if phase == 'train':
                        #total_loss.backward()
                        #optimizer.step()
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    
                batch_size = inputs.size(0)
                running_loss += total_loss.item() * batch_size
                running_root_loss += root_loss.item() * batch_size
                running_tip_loss += tip_loss.item() * batch_size
                running_source_loss += source_loss.item() * batch_size

                if phase == 'test':
                    #FineTune that 
                    preds_roots = torch.sigmoid(outputs['roots']) > 0.5
                    preds_tips = torch.sigmoid(outputs['tips']) > 0.5
                    preds_sources = torch.sigmoid(outputs['sources']) > 0.5
                    #########

                    iou_roots += calculate_iou(preds_roots, roots_masks)
                    iou_tips += calculate_iou(preds_tips, tips_masks)
                    iou_sources += calculate_iou(preds_sources, sources_masks)

                    f1_roots += calculate_f1(preds_roots, roots_masks)
                    f1_tips += calculate_f1(preds_tips, tips_masks)
                    f1_sources += calculate_f1(preds_sources, sources_masks)

                    num_batches += 1
                
            dataset_size = len(dataloaders[phase].dataset)
            epoch_root_loss = running_root_loss / dataset_size
            epoch_tip_loss = running_tip_loss / dataset_size
            epoch_source_loss = running_source_loss / dataset_size

            epoch_loss = epoch_root_loss + epoch_tip_loss + epoch_source_loss
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
                #print images and save models every n epochs
                if (epoch + 1) % 20 == 0:
                    #visualize_outputs(inputs, masks, outputs, num_images=5,experiment_name=experiment_name)
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, f'epoch_model_{experiment_name}_{epoch+1}.pth')    

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

                # Calcola la somma delle F1 score
                f1_sum = f1_roots + f1_tips + f1_sources

                # Early stopping basato sulla somma delle F1 score
                if f1_sum > best_f1_sum:
                    best_f1_sum = f1_sum
                    epochs_no_improve = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, f'best_model_{experiment_name}.pth')
                    best_epoch = epoch + 1
                    best_loss = epoch_loss
                    print('Best model updated.')
                    print(f'Best F1 Sum: {best_f1_sum:.4f}')
                else:
                    epochs_no_improve += 1
                    print(f'Epochs with no improvement: {epochs_no_improve}/{early_stopping_patience}')

                if epochs_no_improve >= early_stopping_patience:
                    print('Early stopping triggered.')
                    break  # Esce dal ciclo delle epoche

                if scheduler:
                    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()

            epoch_duration = time.time() - epoch_start_time
            minutes, seconds = divmod(epoch_duration, 60)
            print(f'epochs time: {int(minutes)},{round(seconds,2)}')
        else:
            continue  # Se non è stato interrotto, continua con la prossima epoca
        break  # Se è stato interrotto, esce dal ciclo delle epoche

    print(f'Training complete. Best val loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    writer.close()
    best_metrics = {
        'epoch': best_epoch,
        'f1_scores': {
            'roots': f1_scores['roots'][-1],
            'tips': f1_scores['tips'][-1],
            'sources': f1_scores['sources'][-1]
        },
        'iou_scores': {
            'roots': iou_scores['roots'][-1],
            'tips': iou_scores['tips'][-1],
            'sources': iou_scores['sources'][-1]
        }
    }
    return model, best_metrics
