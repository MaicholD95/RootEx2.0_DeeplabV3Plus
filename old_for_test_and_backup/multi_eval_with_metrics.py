import torch
import numpy as np
import itertools
from DeepLabV3.model import MultiHeadDeeplabV3Plus
from dataset import CustomRGBDataset
import cv2
import os
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Initialize measurements
iou_measures = []
dice_measures = []
missing_source_measures = []

# Class for prediction without visualization
class Predictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        self.model = MultiHeadDeeplabV3Plus(pretrained_backbone_path=None).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Adjust for DataParallel models
        if 'module.' in list(checkpoint.keys())[0]:
            new_state_dict = {k[7:]: v for k, v in checkpoint.items()}
            checkpoint = new_state_dict
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        self.transform = None
        self.pred_root_threshold = 0.8
        self.pred_tip_threshold = 0.3
        self.pred_source_threshold = 0.3

    def set_transform(self, height, width):
        self.transform = A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def predict(self, image):
        original_height, original_width = image.shape[:2]
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            preds_roots = torch.sigmoid(outputs['roots'])
            preds_tips = torch.sigmoid(outputs['tips'])
            preds_sources = torch.sigmoid(outputs['sources'])
            
            # Resize preds to original image size
            preds_roots = torch.nn.functional.interpolate(
                preds_roots, size=(original_height, original_width), mode='bilinear', align_corners=False)
            preds_tips = torch.nn.functional.interpolate(
                preds_tips, size=(original_height, original_width), mode='bilinear', align_corners=False)
            preds_sources = torch.nn.functional.interpolate(
                preds_sources, size=(original_height, original_width), mode='bilinear', align_corners=False)
            
            # Apply thresholds
            preds_roots = (preds_roots > self.pred_root_threshold).float()
            preds_tips = (preds_tips > self.pred_tip_threshold).float()
            preds_sources = (preds_sources > self.pred_source_threshold).float()
            
            # Convert to numpy arrays
            preds_roots = preds_roots.cpu().numpy()[0, 0, :, :]
            preds_tips = preds_tips.cpu().numpy()[0, 0, :, :]
            preds_sources = preds_sources.cpu().numpy()[0, 0, :, :]
            
        return preds_roots, preds_tips, preds_sources

    def calculate_metrics(self, preds, masks):
        preds_roots, preds_tips, preds_sources = preds
        mask_roots, mask_tips, mask_sources = masks

        # Initialize lists to store metrics
        iou_scores = []
        dice_scores = []
        missing_counts = []

        # Calculate IoU and Dice for Root class (index 0)
        pred = preds_roots
        mask = mask_roots

        pred_flat = pred.flatten()
        mask_flat = mask.flatten()
        intersection = np.sum(pred_flat * mask_flat)
        union = np.sum(pred_flat) + np.sum(mask_flat) - intersection
        iou = intersection / (union + 1e-6)
        dice = 2 * intersection / (np.sum(pred_flat) + np.sum(mask_flat) + 1e-6)

        iou_scores.append(iou)
        dice_scores.append(dice)
        missing_counts.append(None)  # No missing count for roots

        # Calculate missing sources for Source class (index 2)
        pred = preds_sources
        mask = mask_sources

        num_labels_mask, labels_mask = cv2.connectedComponents(mask.astype(np.uint8))
        num_mask_sources = num_labels_mask - 1  # Subtract 1 for the background label

        # For the prediction
        num_labels_pred, labels_pred = cv2.connectedComponents(pred.astype(np.uint8))
        num_pred_sources = num_labels_pred - 1  # Subtract 1 for the background label

        missing_count = num_mask_sources - num_pred_sources  # Difference between real and predicted sources

        missing_counts.append(missing_count)  # Difference between real and predicted sources

        return iou_scores[0], dice_scores[0], np.abs(missing_counts[1])  # Return Root IoU/Dice and Source missing count

# Function to iterate over all combinations and calculate metrics
def run_experiments(predictor, dataset):
    resize_heights =[1400] #range(1000, 1500, 100)
    resize_widths = range(1000, 1500, 100)
    root_thresholds = np.arange(0.1, 0.9, 0.2)
    tip_thresholds = np.arange(0.1, 1.1, 1)
    source_thresholds = np.arange(0.1, 1.1,1)
    results = []

    for height, width, root_thresh, tip_thresh, source_thresh in itertools.product(
            resize_heights, resize_widths, root_thresholds, tip_thresholds, source_thresholds):
        
        #print(f"Testing Resize ({height}, {width}), thresholds: root {root_thresh}, tip {tip_thresh}, source {source_thresh}")
        
        # Set transformations and thresholds
        predictor.set_transform(height, width)
        predictor.pred_root_threshold = root_thresh
        predictor.pred_tip_threshold = tip_thresh
        predictor.pred_source_threshold = source_thresh

        # Metrics for the current configuration
        temp_iou_measures = []
        temp_dice_measures = []
        temp_missing_source_measures = []

        for index in range(len(dataset)):
            annotation = dataset.data[index][0]
            img_name = annotation['root']['name']
            if "\\" in img_name:
                img_name = img_name.split('\\')[-1]
            image_path = os.path.join(dataset.image_dir, img_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create masks at original image size
            masks = dataset._create_rgb_mask(dataset.data[index], image.shape[:2], isTrain=False).transpose(2, 0, 1)
            mask_roots = masks[0]
            mask_tips = masks[1]
            mask_sources = masks[2]
            masks = (mask_roots, mask_tips, mask_sources)
            
            preds = predictor.predict(image)
            iou, dice, missing_source = predictor.calculate_metrics(preds, masks)

            temp_iou_measures.append(iou)
            temp_dice_measures.append(dice)
            temp_missing_source_measures.append(missing_source)

        # Average metrics
        avg_iou = np.mean(temp_iou_measures)
        avg_dice = np.mean(temp_dice_measures)
        avg_missing_source = np.sum(temp_missing_source_measures)

        result_line = (f"Resize: ({height}, {width}), "
                       f"Root Threshold: {root_thresh}, "
                       f"Tip Threshold: {tip_thresh}, "
                       f"Source Threshold: {source_thresh}, "
                       f"Avg IoU Root: {avg_iou:.4f}, "
                       f"Avg Dice Root: {avg_dice:.4f}, "
                       f"Missing Sources: {avg_missing_source}")
        results.append(result_line)

        print(result_line)

    with open("metric_results.txt", "w") as f:
        for line in results:
            f.write(line + "\n")

# Initialize model and dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model_path = 'best_model_Exp_4_Dice_BCE_W_#1_LR_1e-4_ReduceLROnPlateau_2_Weights_0.3_0.7_1_2.pth'  
model_path = 'best_model_Exp_6_Dice_BCE_W_#1_LR_1e-4_ReduceLROnPlateau_2_Weights_0.5_0.5_8_20.pth'
model_path = 'best_model_Exp_7_B20_6_Dice_BCE_W_10_20_#1_LR_1e-4_ReduceLROnPlateau_1_Weights_0.1_0.9_1_2.pth'
model_path ='best_model_Exp_9_B20_6_Dice_BCE_W_10_20_#1_LR_1e-4_ReduceLROnPlateau_1_Weights_0.1_0.9_2_5.pth'

predictor = Predictor(model_path, device=device)

dataset_path = r"D:\WORK\DATA\root\semanticSegm\test"
test_json_files = glob(os.path.join(dataset_path, '*.json'))
test_dataset = CustomRGBDataset(json_files=test_json_files, image_dir=dataset_path, phase='test', isTraining=False)

# Run experiments
run_experiments(predictor, test_dataset)
