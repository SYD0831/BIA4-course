import os
import shutil
import random
import torch
import cv2
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

# KEY: Import required classes and functions from your project modules
# The script must be able to locate the 'model' and 'data' directories
from model.standard_net import StandardNetLightning
from data.mpidb_dataset import to7ch  # Core 7-channel transformation function

def preprocess_single_image(image_path: str, image_size: int = 100):
    """
    Load and preprocess a single new image to match model input requirements.
    This function fully replicates the preprocessing pipeline used during training.

    Args:
        image_path (str): File path of the image to process.
        image_size (int): Target input resolution for the model.

    Returns:
        torch.Tensor: A processed tensor ready for model inference.
    """
    # 1. Read image using OpenCV (BGR format)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Error: Cannot read image file '{image_path}'. Please check the path.")
    
    # 2. Resize to the model-specified size (100×100)
    img_resized = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # 3. Convert BGR image to 7-channel representation (core feature engineering)
    img_7ch = to7ch(img_resized)

    # 4. Convert to PyTorch tensor
    # ToTensorV2 automatically:
    #   - Rearranges axes: (H, W, C) → (C, H, W)
    #   - Converts dtype: numpy.ndarray → torch.float32
    #   - Normalizes pixel values: [0, 255] → [0.0, 1.0]
    tensor_converter = ToTensorV2()
    tensor = tensor_converter(image=img_7ch)['image']

    # 5. Add batch dimension
    # Model input shape is (N, C, H, W); for a single image N = 1
    tensor = tensor.unsqueeze(0)

    return tensor

def predict(model_path: str, image_path: str):
    """
    Load a trained model and perform inference on a single image.

    Args:
        model_path (str): Path to the trained .ckpt model.
        image_path (str): Path of the image to classify.
    """
    print("--- Starting Prediction ---")
    
    # 1. Load model
    print(f"1. Loading model from '{model_path}'...")
    try:
        model = StandardNetLightning.load_from_checkpoint(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please check the path.")
        return

    model.eval()  # Very important!

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   Model loaded and moved to device: {device.type.upper()}")

    # 2. Preprocess input image
    print(f"2. Preprocessing image '{image_path}'...")
    try:
        input_tensor = preprocess_single_image(image_path)
        input_tensor = input_tensor.to(device)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Error during image processing: {e}")
        return
    print("   Image preprocessing completed. Input tensor shape:", input_tensor.shape)

    # 3. Perform inference
    print("3. Running inference...")
    with torch.no_grad():
        logits = model(input_tensor)
    
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1).squeeze()
    
    predicted_index = torch.argmax(probabilities).item()

    # Class index → name mapping (must match training)
    class_map = {0: 'falciparum', 1: 'ovale', 2: 'vivax'}
    predicted_class_name = class_map.get(predicted_index, "Unknown Class")

    confidence = probabilities[predicted_index].item()

    print("\n--- Prediction Result ---")
    print(f"Predicted Class: {predicted_class_name.upper()}")
    print(f"Confidence: {confidence:.2%}")
    print("\nProbability Distribution:")
    for i, prob in enumerate(probabilities):
        class_name = class_map.get(i, f"Class {i}")
        print(f"  - {class_name:<12}: {prob.item():.2%}")

def predict_batch(model_path: str, image_paths: list[Path]) -> pd.DataFrame:
    """
    Load model and predict a batch of images, returning results as a DataFrame.

    Args:
        model_path (str): Path to the trained .ckpt model.
        image_paths (list[Path]): List of image paths.

    Returns:
        pd.DataFrame: Prediction results.
    """
    print("--- Starting Batch Prediction ---")
    
    print(f"1. Loading model from '{model_path}'...")
    try:
        model = StandardNetLightning.load_from_checkpoint(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return pd.DataFrame()

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   Model loaded and moved to device: {device.type.upper()}")

    # Class mapping
    class_map = {0: 'falciparum', 1: 'vivax', 2: 'ovale', 3: 'negative'}
    
    results = []

    print(f"2. Processing {len(image_paths)} images...")
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Batch Predict", unit="image"):
            input_tensor = preprocess_single_image(str(img_path))
            
            if input_tensor is None:
                print(f"Warning: Failed to process image '{img_path.name}'. Skipped.")
                results.append({
                    'filename': img_path.name,
                    'predicted_class': 'Error',
                    'confidence': 0.0,
                    'prob_falciparum': 0.0,
                    'prob_vivax': 0.0,
                    'prob_ovale': 0.0,
                    'prob_negative': 0.0
                })
                continue
            
            input_tensor = input_tensor.to(device)
            logits = model(input_tensor)
            
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu()
            predicted_index = torch.argmax(probabilities).item()
            predicted_class_name = class_map.get(predicted_index, "Unknown Class")
            confidence = probabilities[predicted_index].item()
            
            results.append({
                'filename': img_path.name,
                'predicted_class': predicted_class_name,
                'confidence': confidence,
                'prob_falciparum': probabilities[0].item(),
                'prob_vivax': probabilities[1].item(),
                'prob_ovale': probabilities[2].item(),
                'prob_negative': probabilities[3].item() if len(probabilities) > 3 else 0.0
            })

    print("3. Batch prediction completed. Creating DataFrame...")
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch classification of malaria images using a trained model.")
    parser.add_argument(
        '-c', '--checkpoint', 
        type=str, 
        default="model.ckpt", 
        help="Path to the trained model checkpoint (.ckpt)."
    )
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True, 
        help="Folder path containing images for prediction."
    )
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        default="prediction_results.csv",
        help="Path to save output CSV file. (Default: prediction_results.csv)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Error: Input path '{args.input}' is not a valid directory.")
    else:
        print(f"Collecting images from '{input_path}'...")
        image_files = list(input_path.glob('*.jpg')) + \
                      list(input_path.glob('*.jpeg')) + \
                      list(input_path.glob('*.png'))
        
        if not image_files:
            print("Error: No image files (.jpg, .jpeg, .png) found in the specified folder.")
        else:
            results_df = predict_batch(model_path=args.checkpoint, image_paths=image_files)

            if not results_df.empty:
                print("\n--- Preview of Results ---")
                print(results_df.head())

                try:
                    results_df.to_csv(args.output_csv, index=False)
                    print(f"Results saved to: {args.output_csv}")
                except Exception as e:
                    print(f"Failed to save CSV file: {e}")
