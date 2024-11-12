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
from torch.utils.data import DataLoader
from train_logic import run_training
import csv
from glob import glob
import torch.nn as nn
import multiprocessing

# Import configurations
from configuration import (
    dataset_path,
    pretrained_backbone_path,
    loss_functions,
    optimizer_params_list,
    scheduler_params_list,
    loss_weights_list
)

# Import custom modules
from dataset import CustomRGBDataset

if __name__ == "__main__":
    
    num_cpu_cores = multiprocessing.cpu_count()
    print(f"Available CPU cores: {num_cpu_cores}")
    num_workers = max(1, num_cpu_cores - 5)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_json_files = glob(os.path.join(dataset_path, 'train', '*.json'))
    test_json_files = glob(os.path.join(dataset_path, 'test', '*.json'))
    train_image_dir = os.path.join(dataset_path, 'train')
    test_image_dir = os.path.join(dataset_path, 'test')
    
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
        'train': DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=True, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=6, shuffle=False, drop_last=True, num_workers=num_workers)
    }
    
    # Generate all possible combinations of configurations
    all_configs = list(product(loss_functions, optimizer_params_list, scheduler_params_list, loss_weights_list))
    
    # File to save the results
    results_file = 'training_results.csv'
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Run',
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
    
    # Iterate over configurations
    starting_test_index = 0 
    for idx, (loss_config, optimizer_params, scheduler_params, loss_weights) in enumerate(all_configs):
        experiment_name = f"Exp_{idx + starting_test_index}_{loss_config['name']}_{optimizer_params['name']}_{scheduler_params['name']}_{loss_weights['name']}"
        print(f"\nRun {idx + starting_test_index}/{len(all_configs)}")
        print(f"Configuration: {experiment_name}")
        
        # Configure the training settings
        config = {
            'criteria': loss_config['criteria'],
            'optimizer_params': optimizer_params,
            'scheduler_params': scheduler_params,
            'loss_weights': loss_weights,
            'num_epochs': 250,
            'device': device,
            'pretrained_backbone': pretrained_backbone_path,
            'experiment_name': experiment_name
        }
        
        # Run training
        best_metrics = run_training(config, device, dataloaders)
        
        # Save metrics
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                idx + starting_test_index,
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
