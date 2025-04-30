import os
import cv2
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import torch
import numpy as np
from predictor import Predictor



# Function to load an image and preprocess it
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Function to save prediction results
def save_predictions(predictions, output_dir, base_name, image_path):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual class masks
    for class_name, pred_mask in predictions.items():
        output_path = os.path.join(output_dir, f"{base_name}_{class_name}.png")
        pred_mask = (pred_mask * 255).astype(np.uint8)  # Scale mask to 0-255
        cv2.imwrite(output_path, pred_mask)
        print(f"Saved {class_name} prediction to {output_path}")
    
    # Load the original image
    image = load_image(image_path)
    
    # Create a combined mask with different weights for visibility
    combined_mask = np.zeros_like(predictions["roots"], dtype=np.uint8)

    # Assign colors to different classes (roots: red, tips: green, sources: blue)
    color_mask = np.zeros_like(image, dtype=np.uint8)

    if "roots" in predictions:
        combined_mask = np.maximum(combined_mask, (predictions["roots"] * 255).astype(np.uint8))
        color_mask[:, :, 0] = np.maximum(color_mask[:, :, 0], (predictions["roots"] * 255).astype(np.uint8))  # Red channel

    if "tips" in predictions:
        combined_mask = np.maximum(combined_mask, (predictions["tips"] * 255).astype(np.uint8))
        color_mask[:, :, 1] = np.maximum(color_mask[:, :, 1], (predictions["tips"] * 255).astype(np.uint8))  # Green channel

    if "sources" in predictions:
        combined_mask = np.maximum(combined_mask, (predictions["sources"] * 255).astype(np.uint8))
        color_mask[:, :, 2] = np.maximum(color_mask[:, :, 2], (predictions["sources"] * 255).astype(np.uint8))  # Blue channel

    # Apply the colored mask overlay on the original image
    overlay_result = apply_mask(image, color_mask)

    # Save the overlaid result
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_result, cv2.COLOR_RGB2BGR))
    print(f"Saved overlay prediction to {overlay_path}")


# Apply the mask on the original image and save the result
def apply_mask(image, color_mask, alpha=0.5):
    """
    Overlay a semi-transparent mask on the original image.
    :param image: Original RGB image
    :param color_mask: Colored segmentation mask
    :param alpha: Transparency factor (0 = fully transparent, 1 = fully opaque)
    :return: Image with mask overlay
    """
    blended = cv2.addWeighted(image, 1, color_mask.astype(np.uint8), alpha, 0)
    return blended


if __name__ == "__main__":
    imgs = ['3']#,'image_0133','image_1152','image_1872','image_2473','image_1614']
    # Define paths and parameters
    for img in imgs:
        model_path = r'C:\Users\maich\Desktop\rootex3\best_model_Exp_6_Dice_BCE_W_#1_LR_1e-4_ReduceLROnPlateau_2_Weights_0.5_0.5_8_20.pth'
        image_path = fr"E:\ricerca\cropped_non_resized\{img}.jpg"
        output_dir = r"E:\r2_predicted_images"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Predictor
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

        # Load and predict
        image = load_image(image_path)
        predictions = predictor.predict(image)

        # Save predictions
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_predictions(predictions, output_dir, base_name, image_path)
