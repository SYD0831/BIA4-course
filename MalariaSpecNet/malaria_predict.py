import os
import shutil
import random
import torch
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

from model.standard_net import StandardNetLightning
from data.mpidb_dataset import to7ch

def preprocess_single_image(image_path: str, image_size: int = 100):
    img_bgr = cv2.imread(image_path)
    img_resized = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img_7ch = to7ch(img_resized)
    tensor = ToTensorV2()(image=img_7ch)["image"]
    return tensor.unsqueeze(0)

def malaria_predict(model_path: str, image_paths: list[Path]) -> pd.DataFrame:
    model = StandardNetLightning.load_from_checkpoint(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_map = {0: 'falciparum', 1: 'vivax', 2: 'ovale', 3: 'negative'}
    results = []

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Prediction"):
            input_tensor = preprocess_single_image(str(img_path)).to(device)
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze()
            predicted_index = torch.argmax(probabilities).item()
            predicted_class = class_map.get(predicted_index, "unknown")
            confidence = probabilities[predicted_index].item()
            results.append([predicted_class, confidence])
    return pd.DataFrame(results, columns=["predicted_class", "confidence"])

if __name__ == "__main__":
    model_path = Path("./model.ckpt")
    image_folder = Path("./input")
    image_files = [Path("./input/ovale.tif")]

    # results = malaria_predict(model_path, image_files)
    # print(results.to_string(index=False))
