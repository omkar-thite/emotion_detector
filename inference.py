import os
import time
import sys
import argparse

import torch
from PIL import Image
from utils import *

from dotenv import load_dotenv

load_dotenv() 

checkpoint_path = os.getenv("CHECKPOINT_PATH",  "models/best_modelv5.pth")

def parse_args():
    parser = argparse.ArgumentParser(description="Emotion detection inference")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    return parser.parse_args()


def load_image(image_path):
    try:
        image = Image.open(image_path).convert('L')
        return image
    except FileNotFoundError:
        sys.exit(f"Error: Image not found: {image_path}")
    except Exception as e:
        sys.exit(f"Error opening image: {e}")


def load_model(checkpoint_path, model, device):
    try:
        model, (mean, std) = load_checkpoint(checkpoint_path, model, device)
        return model, (mean, std)
    except FileNotFoundError:
        sys.exit(f"Error: Checkpoint not found: {checkpoint_path}")
    except Exception as e:
        sys.exit(f"Error loading model: {e}")


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = SimpleCNN(num_classes=7).to(device)
    model.eval()
    
    model, (mean, std) = load_model(checkpoint_path, model, device)
    model = torch.compile(model)

    # Load image
    image = load_image(args.image_path)
    image = transform(image, mean, std)
    image = image.to(device)

    # Inference with timing
    start = time.perf_counter()
    
    with torch.inference_mode():
        
        logits = model(image.unsqueeze(0).to(device))
        pred = torch.argmax(logits, dim=1)
    
    end = time.perf_counter()

    print(emotion_map[pred.item()])

    print(f'Inference latency: {end - start:.4f}s')


if __name__ == "__main__":
    main()