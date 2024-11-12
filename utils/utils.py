# utils.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_score,precision_score, recall_score
import random
def calculate_f1(pred, target):
    pred = pred.reshape(-1).cpu().numpy()
    target = target.reshape(-1).cpu().numpy()
    precision = precision_score(target, pred, average='binary', zero_division=1)
    recall = recall_score(target, pred, average='binary', zero_division=1)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
def calculate_iou(pred, target):
    pred = pred.reshape(-1).cpu().numpy()
    target = target.reshape(-1).cpu().numpy()
    return jaccard_score(target, pred, average='binary')

def visualize_outputs(inputs, masks, outputs, num_images=10):
    """
    Visualizes the input image, ground truth masks, and model predictions.
    
    Args:
        inputs (torch.Tensor): Input images tensor of shape [B, 3, H, W]
        masks (torch.Tensor): Ground truth masks tensor of shape [B, 3, H, W]
        outputs (dict): Model outputs containing 'roots', 'tips', 'sources', each of shape [B, 1, H, W]
        num_images (int): Number of images to save from the batch
    """
    for i in range(num_images):
        # Move tensors to CPU and detach
        input_img = inputs[i].cpu().detach()
        mask = masks[i].cpu().detach()
        root_pred = outputs['roots'][i].cpu().detach()
        tip_pred = outputs['tips'][i].cpu().detach()
        source_pred = outputs['sources'][i].cpu().detach()
        
        # Convert input image from [3, H, W] to [H, W, 3]
        input_img = input_img.permute(1, 2, 0).numpy()
        
        # If you applied normalization during preprocessing, denormalize here
        # Example assuming ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        input_img = (input_img * std) + mean
        input_img = np.clip(input_img, 0, 1)
        
        # Ground truth masks: [3, H, W] -> [H, W, 3]
        mask = mask.permute(1, 2, 0).numpy()
        
        # Predicted masks: [1, H, W] -> [H, W]
        root_pred = torch.sigmoid(root_pred) > 0.5
        tip_pred = torch.sigmoid(tip_pred) > 0.3
        source_pred = torch.sigmoid(source_pred) > 0.3
        
        root_pred = root_pred.squeeze().numpy()
        tip_pred = tip_pred.squeeze().numpy()
        source_pred = source_pred.squeeze().numpy()
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Plot Input Image
        axes[0, 0].imshow(input_img)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        # Plot Ground Truth Masks
        axes[0, 1].imshow(mask[:, :, 0], cmap='gray')
        axes[0, 1].set_title('Ground Truth Roots')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(mask[:, :, 1], cmap='gray')
        axes[0, 2].set_title('Ground Truth Tips')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(mask[:, :, 2], cmap='gray')
        axes[0, 3].set_title('Ground Truth Sources')
        axes[0, 3].axis('off')
        
        # Plot Predicted Masks
        axes[1, 0].imshow(input_img)
        axes[1, 0].set_title('Input Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(root_pred, cmap='gray')
        axes[1, 1].set_title('Predicted Roots')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(tip_pred, cmap='gray')
        axes[1, 2].set_title('Predicted Tips')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(source_pred, cmap='gray')
        axes[1, 3].set_title('Predicted Sources')
        axes[1, 3].axis('off')
        
        # Overlay Function
        def overlay_mask(image, mask, color=[1, 0, 0]):
            """
            Overlays a mask on the image.
            
            Args:
                image (numpy.ndarray): Image array of shape [H, W, 3]
                mask (numpy.ndarray): Mask array of shape [H, W]
                color (list): RGB color for the mask overlay
            
            Returns:
                numpy.ndarray: Image with mask overlay
            """
            overlay = image.copy()
            overlay[mask] = color
            return overlay
        
        # Overlay Predicted Masks
        overlay_root = overlay_mask(input_img, root_pred, color=[1, 0, 0])  # Red
        overlay_tip = overlay_mask(input_img, tip_pred, color=[0, 1, 0])    # Green
        overlay_source = overlay_mask(input_img, source_pred, color=[0, 0, 1])  # Blue
        
        axes[2, 0].imshow(input_img)
        axes[2, 0].set_title('Input Image')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(overlay_root)
        axes[2, 1].set_title('Overlay Predicted Roots')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(overlay_tip)
        axes[2, 2].set_title('Overlay Predicted Tips')
        axes[2, 2].axis('off')
        
        axes[2, 3].imshow(overlay_source)
        axes[2, 3].set_title('Overlay Predicted Sources')
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        
        # Save each figure as an image
        plt.savefig(f'tmg_{i}.jpg')
        plt.close(fig)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False