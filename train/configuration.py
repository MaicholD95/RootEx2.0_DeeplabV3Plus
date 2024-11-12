import torch
import torch.nn as nn
from losses.loss import DiceLoss, FocalLoss, TverskyLoss

# Define the dataset and pretrained backbone paths
dataset_path = r"D:\WORK\DATA\root\semantic_segm_deeplabv3+_multiclass_masks\Cropped_512_tip_aug_mixed_Res"
pretrained_backbone_path = 'DeepLabV3\pth\best_deeplabv3_resnet50_voc_os16.pth'


# Loss function configurations
loss_functions = [
    # Uncomment or add configurations as needed
    # {
    #     'criteria': {
    #         'roots': {'dice': DiceLoss(), 'bce': nn.BCEWithLogitsLoss()},
    #         'tips': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20)),
    #         'sources': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(60))
    #     },
    #     'name': 'Dice_BCE_#1'
    # },
    {
        'criteria': {
            'roots': {'dice': DiceLoss(), 'bce': nn.BCEWithLogitsLoss()},
            'tips': nn.BCEWithLogitsLoss(weight=torch.tensor(6)),
            'sources': nn.BCEWithLogitsLoss(weight=torch.tensor(9))
        },
        'name': 'B20_6_Dice_BCE_W_10_20_#1'
    },
    {
        'criteria': {
            'roots': {'dice': DiceLoss(), 'bce': nn.BCEWithLogitsLoss()},
            'tips': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(200)),
            'sources': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(80))
        },
        'name': 'B20_6_Dice_BCE_PosW_#2'
    }
]

# Optimizer parameter configurations
optimizer_params_list = [
    {
        'lr_shared': 1e-4,
        'lr_roots': 1e-4,
        'lr_tips': 1e-4,
        'lr_sources': 1e-4,
        'name': 'LR_1e-4'
    },
    # Uncomment or add configurations as needed
    # {
    #     'lr_shared': 1e-3,
    #     'lr_roots': 1e-3,
    #     'lr_tips': 1e-5,
    #     'lr_sources': 1e-5,
    #     'name': 'LR_1e-3_5'
    # }
]

# Scheduler parameter configurations
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
    # Uncomment or add configurations as needed
    # {
    #     'type': 'StepLR',
    #     'step_size': 25,
    #     'gamma': 0.1,
    #     'name': 'StepLR'
    # }
]

# Loss weights configurations
loss_weights_list = [
    {
        'root_dice': 0.3,
        'root_bce': 0.7,
        'tip_loss': 1,
        'source_loss': 2,
        'name': 'Weights_0.3_0.7_1_2'
    },
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
    }
    # },
    # {
    #     'root_dice': 0.1,
    #     'root_bce': 0.9,
    #     'tip_loss': 2,
    #     'source_loss': 5,
    #     'name': 'Weights_0.1_0.9_2_5'
    # },
]
