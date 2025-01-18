# predictor.py
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import copy
from scipy.optimize import linear_sum_assignment
from DeepLabV3.model import MultiHeadDeeplabV3Plus
import matplotlib.pyplot as plt
import os

class Predictor:
    def __init__(
        self,
        model_path,
        device='cuda',
        resize_height=1400,
        resize_width=1400,
        root_threshold=0.5,
        tip_threshold=0.7,
        source_threshold=0.3,
        sigma=15,
        area_threshold=320,
        circle_radius=20,
        spacing_radius=18
    ):
        self.device = torch.device(device)
        self.model = MultiHeadDeeplabV3Plus(pretrained_backbone_path=None).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle models saved with DataParallel
        if 'module.' in list(checkpoint.keys())[0]:
            # Remove 'module.' prefix
            new_state_dict = {}
            for k, v in checkpoint.items():
                name = k[7:]  # Remove 'module.'
                new_state_dict[name] = v
            checkpoint = new_state_dict
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Save thresholds
        self.root_threshold = root_threshold
        self.tip_threshold = tip_threshold
        self.source_threshold = source_threshold

        # Save additional parameters
        self.sigma = sigma
        self.area_threshold = area_threshold
        self.circle_radius = circle_radius
        self.spacing_radius = spacing_radius

        # Define transformations
        self.transform = A.Compose([
            A.Resize(height=resize_height, width=resize_width),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # Initialize metrics as instance variables
        self.iou_measures = []
        self.dice_measures = []
        self.tip_measures = []
        self.source_measures = []
        self.missing_tips_measures = []
        self.missing_source_measures = []
        self.overestimate_tips_measures = []
        self.overestimate_source_measures = []
        self.weighted_tip_measures = []

        self.total_gt_tips = 0
        self.total_gt_sources = 0

    def predict(self, image):
        H_orig, W_orig = image.shape[:2]
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            # Apply sigmoid to each output
            outputs['roots'] = torch.sigmoid(outputs['roots'])
            outputs['tips'] = torch.sigmoid(outputs['tips'])
            outputs['sources'] = torch.sigmoid(outputs['sources'])

            # Resize outputs to original image size using bilinear interpolation
            outputs['roots'] = torch.nn.functional.interpolate(
                outputs['roots'], size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            outputs['tips'] = torch.nn.functional.interpolate(
                outputs['tips'], size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            outputs['sources'] = torch.nn.functional.interpolate(
                outputs['sources'], size=(H_orig, W_orig), mode='bilinear', align_corners=False)

            # Apply thresholds after resizing
            preds_roots = (outputs['roots'] > self.root_threshold).float()
            preds_tips = (outputs['tips'] > self.tip_threshold).float()
            preds_sources = (outputs['sources'] > self.source_threshold).float()

            # Convert to numpy arrays
            preds_roots = preds_roots.cpu().numpy()[0, 0, :, :]
            preds_tips = preds_tips.cpu().numpy()[0, 0, :, :]
            preds_sources = preds_sources.cpu().numpy()[0, 0, :, :]

            # Store predictions in a dictionary
            preds_resized = {
                'roots': preds_roots,
                'tips': preds_tips,
                'sources': preds_sources
            }

        return preds_resized

    def visualize(self, image, preds, masks=None, name="", gt_tips_center=[], gt_source_center=[]):
        # Calculate metrics
        class_names = ['roots', 'tips', 'sources']
        if masks is not None:
            iou_scores, dice_scores, distance_scores, missing_counts, overestimate_counts, weighted_distance_scores = self.calculate_metrics(
                preds, masks, gt_tips_center=gt_tips_center, gt_source_center=gt_source_center)
            # Print metrics
            for class_name in class_names:
                if class_name == 'roots':
                    print(f"{class_name} - IoU: {iou_scores[class_name]:.4f}")
                    print(f" Dice: {dice_scores[class_name]:.4f}")
                else:
                    if distance_scores[class_name]:
                        avg_distance = np.mean(distance_scores[class_name])
                        print(f"{class_name} - Average Normalized Distance: {avg_distance:.4f}")
                    else:
                        print(f"{class_name} - No matches found")
                    print(f" Missing {class_name}: {missing_counts[class_name]}")
                    print(f" Overestimated {class_name}: {overestimate_counts[class_name]}")
                    if class_name == 'tips' and weighted_distance_scores[class_name]:
                        self.weighted_tip_measures.append(np.mean(weighted_distance_scores[class_name]))
                        print(f"Weighted distance {class_name}: {np.mean(weighted_distance_scores[class_name])}")
        else:
            iou_scores = {name: None for name in class_names}
            dice_scores = {name: None for name in class_names}
            distance_scores = {name: None for name in class_names}
            missing_counts = {name: None for name in class_names}
            overestimate_counts = {name: None for name in class_names}

        # Update global metrics
        self.iou_measures.append(iou_scores['roots'])
        self.dice_measures.append(dice_scores['roots'])
        self.tip_measures.append(np.mean(distance_scores['tips']) if distance_scores['tips'] else 0)
        self.source_measures.append(np.mean(distance_scores['sources']) if distance_scores['sources'] else 0)
        self.missing_tips_measures.append(missing_counts['tips'])
        self.missing_source_measures.append(missing_counts['sources'])
        self.overestimate_tips_measures.append(overestimate_counts['tips'])
        self.overestimate_source_measures.append(overestimate_counts['sources'])

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)

        # First row: original masks
        if masks is not None:
            for idx, class_name in enumerate(class_names):
                mask = masks[class_name]
                overlaid = self.overlay_mask_on_image(image, mask, color=(0, 255, 0))
                axes[0, idx].imshow(overlaid)
                axes[0, idx].set_title(f'Original {class_name}')

        # Second row: predicted masks
        for idx, class_name in enumerate(class_names):
            pred_mask = preds[class_name]
            if pred_mask.dtype != np.uint8:
                pred_mask = (pred_mask > 0.5).astype(np.uint8)
            if class_name == 'roots':
                overlaid = self.overlay_mask_on_image(image, pred_mask, color=(0, 255, 0))
                title = f'Predicted {class_name} overlaid'
                if iou_scores[class_name] is not None and dice_scores[class_name] is not None:
                    title += f'\nIoU: {iou_scores[class_name]:.4f}, Dice: {dice_scores[class_name]:.4f}'
                axes[1, idx].imshow(overlaid)
                axes[1, idx].set_title(title)
            else:
                tips_selected = (class_name == 'tips')
                overlaid = self.overlay_mask_on_image(image, pred_mask, color=(0, 0, 255))
                centers = self.compute_enclosing_circle_centers(pred_mask, tips_selected)
                for center in centers:
                    cv2.circle(overlaid, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
                    cv2.drawMarker(overlaid, (int(center[0]), int(center[1])), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                title = f'Predicted {class_name}'
                if distance_scores[class_name]:
                    avg_distance = np.mean(distance_scores[class_name])
                    title += f'\nAvg Norm Dist: {avg_distance:.4f}'
                else:
                    title += '\nNo matches found'
                title += f'\nMissing: {missing_counts[class_name]}, Overestimated: {overestimate_counts[class_name]}'
                axes[1, idx].imshow(overlaid)
                axes[1, idx].set_title(title)

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        # Ensure the directory exists
        output_dir = 'predicted_imgs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, name.replace('jpg', 'post_processed.jpg')))
        plt.close(fig)  # Close the figure to free memory

    def overlay_mask_on_image(self, image, mask, color=(255, 0, 0), alpha=0.5):
        mask_bin = (mask > 0.5).astype(np.uint8)
        mask_color = np.zeros_like(image)
        mask_color[mask_bin == 1] = color
        overlaid = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)
        return overlaid

    def compute_enclosing_circle_centers(self, mask, tips_selected=False):
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            (cX, cY), radius_contour = cv2.minEnclosingCircle(cnt)
            cX, cY = int(cX), int(cY)
            if area < 10:
                continue
            if area > self.area_threshold and tips_selected:
                x, y, w, h = cv2.boundingRect(cnt)
                # Calculate number of circles based on customizable circle_radius
                number_of_circles = int(np.ceil(area / (np.pi * (self.circle_radius / 2) ** 2)))
                # Calculate spacing based on customizable spacing_radius
                spacing = int(2 * (self.spacing_radius / 2))
                if h > w:
                    start_y = cY - (number_of_circles // 2) * spacing
                    for i in range(number_of_circles):
                        new_cY = start_y + i * spacing
                        centers.append((cX, new_cY))
                else:
                    start_x = cX - (number_of_circles // 2) * spacing
                    for i in range(number_of_circles):
                        new_cX = start_x + i * spacing
                        centers.append((new_cX, cY))
            else:
                (x_center, y_center), radius = cv2.minEnclosingCircle(cnt)
                centers.append((x_center, y_center))
        return centers

    def calculate_metrics(self, preds, masks, epsilon=1e-6, gt_tips_center=[], gt_source_center=[]):
        class_names = ['roots', 'tips', 'sources']

        # Initialize metric dictionaries
        iou_scores = {}
        dice_scores = {}
        distance_scores = {}
        weighted_distance_scores = {}
        missing_counts = {}
        overestimate_counts = {}

        for class_name in class_names:
            pred = preds[class_name]
            mask = masks[class_name]

            if class_name == 'roots':  # Root class
                pred_flat = pred.flatten()
                mask_flat = mask.flatten()
                intersection = np.sum(pred_flat * mask_flat)
                union = np.sum(pred_flat) + np.sum(mask_flat) - intersection
                iou = (intersection + epsilon) / (union + epsilon)
                dice = (2 * intersection + epsilon) / (np.sum(pred_flat) + np.sum(mask_flat) + epsilon)
                iou_scores[class_name] = iou
                dice_scores[class_name] = dice
                distance_scores[class_name] = None
                weighted_distance_scores[class_name] = None
                missing_counts[class_name] = None
                overestimate_counts[class_name] = None
            else:  # Tips and Sources
                tips_selected = (class_name == 'tips')
                selected_center = gt_tips_center if tips_selected else gt_source_center
                pred_centers = self.compute_enclosing_circle_centers(pred, tips_selected)
                if class_name == 'tips':
                    self.total_gt_tips += len(gt_tips_center)
                elif class_name == 'sources':
                    self.total_gt_sources += len(gt_source_center)

                if not pred_centers and not selected_center:
                    distances = []
                    missing_count = 0
                    overestimate_count = 0
                elif not pred_centers:
                    distances = []
                    missing_count = len(selected_center)
                    overestimate_count = 0
                elif not selected_center:
                    distances = []
                    missing_count = 0
                    overestimate_count = len(pred_centers)
                else:
                    cost_matrix = np.zeros((len(selected_center), len(pred_centers)))
                    for m_idx, m_center in enumerate(selected_center):
                        for p_idx, p_center in enumerate(pred_centers):
                            distance = np.linalg.norm(np.array(m_center) - np.array(p_center))
                            cost_matrix[m_idx, p_idx] = distance
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    distances = []
                    MISSING_W = self.sigma / 2
                    missing_count = 0
                    overestimate_count = 0

                    for m_idx, p_idx in zip(row_ind, col_ind):
                        norm_distance = cost_matrix[m_idx, p_idx] / self.sigma
                        if norm_distance <= MISSING_W / 2:
                            distances.append(norm_distance)
                        else:
                            missing_count += 1
                            overestimate_count += 1
                            
                    mask_indices = set(range(len(selected_center)))
                    pred_indices = set(range(len(pred_centers)))
                    unmatched_mask_indices = mask_indices - set(row_ind)
                    unmatched_pred_indices = pred_indices - set(col_ind)
                    missing_count += len(unmatched_mask_indices)
                    overestimate_count += len(unmatched_pred_indices)
                    if class_name == 'tips':
                        weighted_distance_scores[class_name] = copy.deepcopy(distances)
                        for _ in range(missing_count):
                            weighted_distance_scores[class_name].append(MISSING_W)
                        for _ in range(overestimate_count):
                            weighted_distance_scores[class_name].append(MISSING_W)
                    else:
                        weighted_distance_scores[class_name] = None
                iou_scores[class_name] = None
                dice_scores[class_name] = None
                distance_scores[class_name] = distances
                missing_counts[class_name] = missing_count
                overestimate_counts[class_name] = overestimate_count

        return iou_scores, dice_scores, distance_scores, missing_counts, overestimate_counts, weighted_distance_scores
