# main.py
import torch
import numpy as np
import cv2
import os
from glob import glob
import sys
# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from predictor import Predictor
from dataset import CustomRGBDataset

def predict_and_visualize(predictor, dataset, index):
    annotation = dataset.data[index][0]
    img_name = annotation['root']['name'].replace('rsml','jpg')
    if "\\" in img_name or "/" in img_name:
        img_name = os.path.basename(img_name)
    image_path = os.path.join(dataset.image_dir, img_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_tip_centers = []
    gt_source_centers = []
    for root in dataset.data[index]:
        if 'tip' in root and root['tip']:
            # for tip in root['tip']:
                #if (tip['x'], tip['y']) not in gt_tip_centers:
                    #gt_tip_centers.append((tip['x'], tip['y']))
                if (root['tip']['x'], root['tip']['y']) not in gt_tip_centers:
                    gt_tip_centers.append((root['tip']['x'], root['tip']['y']))
                    
           # gt_tip_centers.append((root['tip']['x'], root['tip']['y']))
        if 'source' in root and root['source']:
            if (root['source']['x'], root['source']['y']) not in gt_source_centers:
                gt_source_centers.append((root['source']['x'], root['source']['y']))
                
    masks_array = dataset._create_rgb_mask(dataset.data[index], image.shape[:2])
    masks_array = masks_array.transpose(2, 0, 1)

    # Convert masks to dictionary
    class_names = ['roots', 'tips', 'sources']
    masks = {class_name: masks_array[idx] for idx, class_name in enumerate(class_names)}

    preds = predictor.predict(image)

    predictor.visualize(image, preds, masks, img_name, gt_tip_centers, gt_source_centers)
    #predictor.visualize_postprocessing_steps(image,img_name,gt_tip_centers)
    predictor.visualize_prediction_vs_gt(image,masks['roots'],preds['roots'],img_name)
    print(f"Image: {img_name}")
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model path
    model_path = r'C:\Users\maich\Desktop\rootex3\best_model_Exp_6_Dice_BCE_W_#1_LR_1e-4_ReduceLROnPlateau_2_Weights_0.5_0.5_8_20.pth'
    #model_path = r'best101_OSR.pth'
    # Instantiate Predictor with desired thresholds and parameters
    predictor = Predictor(
        model_path,
        device=device,
        resize_height=1400,
        resize_width=1400,
        root_threshold=0.5,
        tip_threshold=0.52,
        source_threshold=0.3,
        sigma=15,            # Custom sigma value
        area_threshold=320,  # Custom area_threshold (320)
        circle_radius=20,    # Custom circle_radius for number_of_circles calculation
        spacing_radius=18    # Custom spacing_radius for spacing calculation
    )

    # Load dataset
    #dataset_path = r"C:\Users\maich\Desktop\rootnav_to_R3_dataset\OSR_Challenge_DB\test"
    dataset_path = r"C:\Users\maich\Desktop\rootex_Paper\RSML\full_test_imgs"
    test_json_files = glob(os.path.join(dataset_path, '*.json'))
    test_dataset = CustomRGBDataset(json_files=test_json_files, image_dir=dataset_path, phase='test', isTraining=False)

    for index in range(len(test_dataset)):
        print(f"\nProcessing image {index+1}/{len(test_dataset)}")
        predict_and_visualize(predictor, test_dataset, index)

    # After processing all images, print metrics
    print(f"iou_mean {np.mean(predictor.iou_measures)}")
    print(f"dice_mean {np.mean(predictor.dice_measures)}")
    print(f"tip_mean_distance {np.mean(predictor.tip_measures)}")
    print(f"source_mean_distance {np.mean(predictor.source_measures)}")
    print(f"missing_tips {np.sum(predictor.missing_tips_measures)}")
    print(f"missing_source {np.sum(predictor.missing_source_measures)}")
    print(f"overestimated_tips {np.sum(predictor.overestimate_tips_measures)}")
    print(f"overestimated_source {np.sum(predictor.overestimate_source_measures)}")
    print(f"weighted tips distance {np.mean(predictor.weighted_tip_measures)}")

    print(f"Total ground truth tips: {predictor.total_gt_tips}")
    print(f"Total ground truth sources: {predictor.total_gt_sources}")
