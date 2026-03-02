#!/usr/bin/env python3
"""
Demo script to show the weather model working without authentication.
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

class SimpleWeatherNet(nn.Module):
    """Simple CNN for weather forecasting"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def demo_weather_prediction():
    """Demonstrate weather prediction with the trained model."""
    
    print("AtmosGen Weather Forecasting Demo")
    print("=" * 40)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleWeatherNet().to(device)
    
    # Load trained weights
    checkpoint_path = "../checkpoints/atmosgen_simple_v1.pth"
    if not os.path.exists(checkpoint_path):
        print("Error: Model checkpoint not found!")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Load and process test image
    test_image_paths = [
        "../core_model/data/satellite_images/img_0000.png",
        "../core_model/data/satellite_images/img_0001.png",
        "../core_model/data/satellite_images/img_0002.png"
    ]
    
    # Find available test image
    test_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print("Error: No test images found!")
        return
    
    print(f"Using test image: {test_image_path}")
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate prediction
    print("Generating weather prediction...")
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Input image
    input_img = input_tensor[0].cpu()
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    input_img = input_img * std + mean
    input_img = torch.clamp(input_img, 0, 1)
    input_img = input_img.permute(1, 2, 0)
    
    axes[0].imshow(input_img)
    axes[0].set_title("Input Weather Image")
    axes[0].axis('off')
    
    # Prediction
    pred_img = prediction[0].cpu().permute(1, 2, 0)
    pred_img = torch.clamp(pred_img, 0, 1)
    
    axes[1].imshow(pred_img)
    axes[1].set_title("Predicted Weather")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("weather_demo_result.png", dpi=150, bbox_inches='tight')
    print("Demo result saved as: weather_demo_result.png")
    
    # Model info
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print("\nModel Information:")
        print(f"  Name: {config.get('model_name', 'Unknown')}")
        print(f"  Training samples: {config.get('training_samples', 'Unknown')}")
        print(f"  Final loss: {config.get('final_loss', 'Unknown')}")
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")
    
    print("\nDemo completed successfully!")
    print("The model is predicting weather patterns from satellite imagery.")

if __name__ == "__main__":
    demo_weather_prediction()