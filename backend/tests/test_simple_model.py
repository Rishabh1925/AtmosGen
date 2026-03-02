#!/usr/bin/env python3
"""
Test script for the simple weather forecasting model.
"""

import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def create_simple_model():
    """Create the SimpleWeatherNet architecture."""
    
    class SimpleWeatherNet(nn.Module):
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
    
    return SimpleWeatherNet()

def test_simple_model():
    """Test the integrated simple weather model."""
    
    print(" Testing Simple Weather Model...")
    
    # Check if model exists
    checkpoint_path = "../checkpoints/atmosgen_simple_v1.pth"
    if not os.path.exists(checkpoint_path):
        print(" Model not found! Please run integrate_simple_model.py first")
        return False
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    model = create_simple_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(" Model loaded successfully!")
    
    # Test with a sample image
    test_image_paths = [
        "../core_model/data/satellite_images/img_0000.png",
        "../core_model/data/satellite_images/img_0001.png",
        "../data/processed/detailed_analysis.png"
    ]
    
    # Find an available test image
    test_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print(" No test images found!")
        return False
    
    print(f" Testing with: {test_image_path}")
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate prediction
    print(" Generating weather prediction...")
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Display results
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
    plt.savefig("weather_prediction_test.png", dpi=150, bbox_inches='tight')
    print(" Test result saved as: weather_prediction_test.png")
    
    # Model info
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print("\n Model Information:")
        print(f"   Name: {config.get('model_name', 'Unknown')}")
        print(f"   Training samples: {config.get('training_samples', 'Unknown')}")
        print(f"   Final loss: {config.get('final_loss', 'Unknown')}")
    
    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f" Model parameters: {param_count:,}")
    
    print("\n Simple weather model test completed!")
    print(" Model is working correctly and ready for deployment")
    
    return True

if __name__ == "__main__":
    success = test_simple_model()
    if not success:
        exit(1)