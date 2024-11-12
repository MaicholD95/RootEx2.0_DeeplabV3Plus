import os
import json
import numpy as np
from skimage import io
from torch.utils.data import Dataset
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def _create_heatmap(shape, points, sigma=2):
    heatmap = np.zeros(shape, dtype=np.float32)
    for x, y in points:
        heatmap = cv2.circle(heatmap, (int(x), int(y)), sigma, 1, -1)
    return heatmap

class CustomRGBDataset(Dataset):
    def __init__(self, json_files, image_dir, phase="train",isTraining = False):
        self.data = []
        self.image_dir = image_dir
        self.isTraining = isTraining
        self.transform = get_transform(phase)

        for json_file in json_files:
            with open(json_file) as f:
                annotations = json.load(f)
                if not annotations:
                    annotations = [{
                        "root": {
                            "name": os.path.basename(json_file).replace('.json', '.jpg'),
                            "points": {
                                "all_points_x": [],
                                "all_points_y": []
                            },
                            "category_id": 0
                        }
                    }]
                self.data.append(annotations)

        #self.transform = self.get_transform(transform)
        print(f"Loaded {len(self.data)} annotations from {len(json_files)} JSON files for directory {self.image_dir}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        annotation = self.data[idx][0]
        img_name = annotation['root']['name']
        if "\\" in img_name:
            img_name = img_name.split('\\')[-1]
        image_path = os.path.join(self.image_dir, img_name)
        image = io.imread(image_path)
        
        masks = self._create_rgb_mask(self.data[idx], image.shape[:2],isTrain=self.isTraining)
        
        if self.transform:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image'].float()
            masks = augmented['mask'].float()  # Maschera RGB
            
            # Permuta le dimensioni se necessario
            if masks.ndimension() == 3 and masks.shape[0] != 3:
                masks = masks.permute(2, 0, 1)  # Da [H, W, C] a [C, H, W]
        
        return image, masks
    def _create_rgb_mask(self, annotations, shape, box_size=15,isTrain=False):
        """
        Generates a mask with independent channels for root, tip, and source.
        Bounding boxes are used for tips and sources to indicate localization regions.
        
        Args:
            annotations (list): List of annotation data.
            shape (tuple): Shape of the mask (height, width).
            box_size (int): Size of the bounding box for tips and sources.
            
        Returns:
            np.ndarray: Mask with three channels for roots, tips, and sources.
        """
        # Initialize an empty mask with three channels
        mask = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
        
        class_to_channel = {
            'root': 0,    # Channel 0 for roots
            'tip': 1,     # Channel 1 for tips
            'source': 2   # Channel 2 for sources
        }
        
        # Generate the root mask with precise points
        for annotation in annotations:
            if 'root' in annotation:
                points_x = annotation['root']['points']['all_points_x']
                points_y = annotation['root']['points']['all_points_y']
                root_points = list(zip(points_x, points_y))
                if root_points:
                    heatmap = _create_heatmap((shape[0], shape[1]), root_points, sigma=5)
                    mask[:, :, class_to_channel['root']] = np.maximum(mask[:, :, class_to_channel['root']], heatmap)
        
        # Generate the tip mask using bounding boxes
        for annotation in annotations:
            if 'tip' in annotation:
                if isTrain:
                    for elem in annotation['tip']:
                        x, y = elem['x'], elem['y']
                        x_min = max(0, int(x - box_size / 2))
                        x_max = min(shape[1], int(x + box_size / 2))
                        y_min = max(0, int(y - box_size / 2))
                        y_max = min(shape[0], int(y + box_size / 2))
                        # Draw the bounding box on the tip channel
                        mask[y_min:y_max, x_min:x_max, class_to_channel['tip']] = 1.0

                else:
                    x= annotation['tip']['x']
                    y= annotation['tip']['y']
                    x_min = max(0, int(x - box_size / 2))
                    x_max = min(shape[1], int(x + box_size / 2))
                    y_min = max(0, int(y - box_size / 2))
                    y_max = min(shape[0], int(y + box_size / 2))
                    # Draw the bounding box on the tip channel
                    mask[y_min:y_max, x_min:x_max, class_to_channel['tip']] = 1.0
        
        # Generate the source mask using bounding boxes
        for annotation in annotations:
            if 'source' in annotation:
                if isTrain:
                    for elem in annotation['source']:
                        x, y = elem['x'], elem['y']
                   
                        # Calculate bounding box coordinates
                        x_min = max(0, int(x - box_size / 2))
                        x_max = min(shape[1], int(x + box_size / 2))
                        y_min = max(0, int(y - box_size / 2))
                        y_max = min(shape[0], int(y + box_size / 2))
                        # Draw the bounding box on the source channel
                        mask[y_min:y_max, x_min:x_max, class_to_channel['source']] = 1.0
                else:
                    x= annotation['source']['x']
                    y= annotation['source']['y']
                    # Calculate bounding box coordinates
                    x_min = max(0, int(x - box_size / 2))
                    x_max = min(shape[1], int(x + box_size / 2))
                    y_min = max(0, int(y - box_size / 2))
                    y_max = min(shape[0], int(y + box_size / 2))
                    # Draw the bounding box on the source channel
                    mask[y_min:y_max, x_min:x_max, class_to_channel['source']] = 1.0
        
        return mask
def get_transform(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(height=512, width=512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.5),
            # A.OneOf([
            #     A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1),
            #     A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            #     A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1),
            # ], p=0.5),
            #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            #A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=0.5),
            #A.CoarseDropout(max_holes=3, max_height=16, max_width=16, fill_value=0, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    #