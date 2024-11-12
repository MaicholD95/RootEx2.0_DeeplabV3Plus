import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
import random
from itertools import product
from dataset import CustomRGBDataset
from DeepLabV3.model import MultiHeadDeeplabV3Plus
from losses.loss import DiceLoss, FocalLoss, CombinedLovaszBCELoss, TverskyLoss
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from train.train_logic import train_model
import csv
from glob import glob
import torch.nn as nn
import multiprocessing


def  run_training(config, device, dataloaders):
    #resnet50 or resnet101
    model = MultiHeadDeeplabV3Plus(pretrained_backbone_path=config['pretrained_backbone'],backbone='resnet50').to(device)
    # Configura l'ottimizzatore
    optimizer = optim.Adam([
        {'params': model.get_shared_parameters(), 'lr': config['optimizer_params']['lr_shared']},
        {'params': model.get_root_parameters(), 'lr': config['optimizer_params']['lr_roots']},
        {'params': model.get_tip_parameters(), 'lr': config['optimizer_params']['lr_tips']},
        {'params': model.get_source_parameters(), 'lr': config['optimizer_params']['lr_sources']}
    ])
    
    # Configura lo scheduler
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

    # Esegui il training
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

if __name__ == "__main__":
    
    num_cpu_cores = multiprocessing.cpu_count()
    print(f"Numero di core CPU disponibili: {num_cpu_cores}")
    num_workers=num_cpu_cores-5
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carica i dati
    dataset_path = r"D:\WORK\DATA\root\semantic_segm_deeplabv3+_multiclass_masks\Cropped_512_tip_aug_mixed_Res"
    train_json_files = glob(dataset_path + '\\train\\*.json')
    test_json_files = glob(dataset_path + '\\test\\*.json')
    
    train_image_dir = dataset_path + '\\train\\'
    test_image_dir = dataset_path + '\\test\\'
    train_dataset = CustomRGBDataset(
        json_files=train_json_files,
        image_dir=train_image_dir,
        phase='train',
        isTraining=True
    )
    test_dataset = CustomRGBDataset(
        json_files=test_json_files,
        image_dir=test_image_dir,
        phase='test',
        isTraining=True
    )

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=True),
        'test': DataLoader(test_dataset, batch_size=6, shuffle=False, drop_last=True)
    }
    
    # Definisci le configurazioni degli iperparametri
    loss_functions = [
        # {
        #     'criteria': {
        #         'roots': {'dice': DiceLoss(), 'bce': nn.BCEWithLogitsLoss()},
        #         'tips': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20)),
        #         'sources': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(60))
        #     },
        #     'name': 'Dice_BCE_#1'
        # },
        # {
        #     'criteria': {
        #         'roots': {'dice': DiceLoss(), 'bce': nn.BCEWithLogitsLoss()},
        #         'tips': nn.BCEWithLogitsLoss(weight=torch.tensor(6)),
        #         'sources': nn.BCEWithLogitsLoss(weight=torch.tensor(9))
        #     },
        #     'name': 'B20_6_Dice_BCE_W_10_20_#1'
        # }
        # ,
        {
            'criteria': {
                'roots': {'dice': DiceLoss(), 'bce': nn.BCEWithLogitsLoss()},
                'tips': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(200)),
                'sources': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(80))
            },
            'name': 'B20_6_Dice_BCE_PosW_#2'
        }
    ]


    optimizer_params_list = [
        {
            'lr_shared': 1e-4,
            'lr_roots': 1e-4,
            'lr_tips': 1e-4,
            'lr_sources': 1e-4,
            'name': 'LR_1e-4'
        }
        ,
        # {
        #     'lr_shared': 1e-3,
        #     'lr_roots': 1e-3,
        #     'lr_tips': 1e-5,
        #     'lr_sources': 1e-5,
        #     'name': 'LR_1e-3_5'
        # }
    
    ]

    scheduler_params_list = [
        {
            'type': 'ReduceLROnPlateau',
            'factor': 0.5,
            'patience': 15,
            'name': 'ReduceLROnPlateau_1'
        },
        {
            'type': 'ReduceLROnPlateau',
            'factor': 0.5,
            'patience': 10,
            'name': 'ReduceLROnPlateau_2'
        },
        # {
        #     'type': 'StepLR',
        #     'step_size': 25,
        #     'gamma': 0.1,
        #     'name': 'StepLR'
        # }
    ]

    # Configurazioni dei pesi delle perdite
    loss_weights_list = [
        # {
        #     'root_dice': 0.3,
        #     'root_bce': 0.7,
        #     'tip_loss': 1,
        #     'source_loss': 2,
        #     'name': 'Weights_0.3_0.7_1_2'
        # },
        {
            'root_dice': 0.1,
            'root_bce': 0.9,
            'tip_loss': 1,
            'source_loss': 2,
            'name': 'Weights_0.1_0.9_1_2'
        },
        {
            'root_dice': 0.2,
            'root_bce': 0.8,
            'tip_loss': 2,
            'source_loss': 3,
            'name': 'Weights_0.2_0.8_2_3'
        },
        {
            'root_dice': 0.1,
            'root_bce': 0.9,
            'tip_loss': 2,
            'source_loss': 5,
            'name': 'Weights_0.1_0.9_2_5'
        },
    ]

 # Genera tutte le combinazioni possibili
    all_configs = list(product(loss_functions, optimizer_params_list, scheduler_params_list, loss_weights_list))
    
    # File per salvare i risultati
    results_file = 'training_results.csv'
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Esecuzione',
            'Experiment Name',
            'Criteria',
            'Optimizer Params',
            'Scheduler Params',
            'Loss Weights',
            'Best Epoch',
            'F1_Roots',
            'F1_Tips',
            'F1_Sources',
            'IoU_Roots',
            'IoU_Tips',
            'IoU_Sources'
        ])
    
    # Itera sulle configurazioni
    test_attuale = 11
    for idx, (loss_config, optimizer_params, scheduler_params, loss_weights) in enumerate(all_configs):
        experiment_name = f"Exp_{idx+test_attuale}_{loss_config['name']}_{optimizer_params['name']}_{scheduler_params['name']}_{loss_weights['name']}"
        print(f"\nEsecuzione {idx+test_attuale}/{len(all_configs)}")
        print(f"Configurazione: {experiment_name}")
        
        # Configura il dizionario di configurazione per il training
        config = {
            'criteria': loss_config['criteria'],
            'optimizer_params': optimizer_params,
            'scheduler_params': scheduler_params,
            'loss_weights': loss_weights,
            'num_epochs': 250,
            'device': device,
            'pretrained_backbone': r'D:\WORK\DATA\root\semantic_segm_multi_head_3class\best_deeplabv3_resnet50_voc_os16.pth',
            'experiment_name': experiment_name
        }
        
        # Esegui il training
        best_metrics = run_training(config, device, dataloaders)
        
        # Salva le metriche
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                idx+1,
                experiment_name,
                loss_config['name'],
                optimizer_params['name'],
                scheduler_params['name'],
                loss_weights['name'],
                best_metrics['epoch'],
                best_metrics['f1_scores']['roots'],
                best_metrics['f1_scores']['tips'],
                best_metrics['f1_scores']['sources'],
                best_metrics['iou_scores']['roots'],
                best_metrics['iou_scores']['tips'],
                best_metrics['iou_scores']['sources']
            ])