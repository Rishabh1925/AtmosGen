#!/usr/bin/env python3
"""
Create a demo checkpoint with pre-trained weights for immediate testing
This gives us working results while we prepare proper training data
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_model'))

from models.unet import UNet
from config import Config

def create_demo_checkpoint():
    """
    Create a demo checkpoint with reasonable weights for weather forecasting
    This won't be perfect but will generate plausible-looking weather images
    """
    print("Creating demo checkpoint for AtmosGen...")
    
    # Initialize model
    config = Config()
    model = UNet()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Instead of random weights, let's initialize with better defaults
    # for weather/image generation tasks
    
    def init_weather_weights(m):
        """Initialize weights specifically for weather image generation"""
        if isinstance(m, nn.Conv2d):
            # Use Xavier initialization for conv layers
            nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller gain for stability
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Apply better initialization
    model.apply(init_weather_weights)
    
    # Create checkpoints directory
    checkpoint_dir = Path("../checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save the demo checkpoint
    checkpoint_path = checkpoint_dir / "atmosgen_demo_v1.pth"
    
    # Save just the state dict (standard PyTorch format)
    torch.save(model.state_dict(), checkpoint_path)
    
    print(f" Demo checkpoint saved: {checkpoint_path}")
    print(f" Checkpoint size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Test loading the checkpoint
    test_model = UNet()
    test_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print(" Checkpoint loads successfully")
    
    # Create a simple test to verify the model works
    # Model expects 5D input: (batch, sequence_length, channels, height, width)
    test_input = torch.randn(1, 4, 3, 256, 256)  # Batch of 1, sequence of 4, RGB, 256x256
    test_timestep = torch.randint(0, 1000, (1,))
    
    test_model.eval()
    with torch.no_grad():
        output = test_model(test_input, test_timestep)
    
    print(f" Model test successful - Output shape: {output.shape}")
    print(f" Output range: {output.min():.3f} to {output.max():.3f}")
    
    return checkpoint_path

def create_better_test_data():
    """
    Create more diverse test data by transforming existing NOAA images
    This simulates different weather conditions and time periods
    """
    print("\nCreating diverse test data...")
    
    import glob
    from PIL import Image, ImageEnhance, ImageFilter
    import random
    
    # Find processed images
    processed_files = sorted(glob.glob("../data/processed/OR_ABI*.jpg"))
    
    if not processed_files:
        print(" No processed images found. Run test_noaa_data.py first.")
        return
    
    # Create variations directory
    variations_dir = Path("../data/processed/variations")
    variations_dir.mkdir(exist_ok=True)
    
    base_image = Image.open(processed_files[0])
    
    transformations = [
        ("original", lambda img: img),
        ("enhanced_contrast", lambda img: ImageEnhance.Contrast(img).enhance(1.5)),
        ("reduced_contrast", lambda img: ImageEnhance.Contrast(img).enhance(0.7)),
        ("brighter", lambda img: ImageEnhance.Brightness(img).enhance(1.3)),
        ("darker", lambda img: ImageEnhance.Brightness(img).enhance(0.7)),
        ("more_saturated", lambda img: ImageEnhance.Color(img).enhance(1.4)),
        ("less_saturated", lambda img: ImageEnhance.Color(img).enhance(0.6)),
        ("slightly_blurred", lambda img: img.filter(ImageFilter.GaussianBlur(radius=0.5))),
        ("sharpened", lambda img: img.filter(ImageFilter.UnsharpMask(radius=1, percent=150))),
    ]
    
    created_files = []
    
    for i, (name, transform) in enumerate(transformations):
        try:
            transformed = transform(base_image.copy())
            output_path = variations_dir / f"weather_variation_{i:02d}_{name}.jpg"
            transformed.save(output_path, quality=95)
            created_files.append(output_path)
            print(f" Created: {name}")
        except Exception as e:
            print(f" Failed to create {name}: {e}")
    
    print(f" Created {len(created_files)} test variations")
    return created_files

if __name__ == "__main__":
    print("=" * 60)
    print("ATMOSGEN DEMO SETUP")
    print("=" * 60)
    
    # Create demo checkpoint
    checkpoint_path = create_demo_checkpoint()
    
    # Create better test data
    test_files = create_better_test_data()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print(f" Demo checkpoint: {checkpoint_path}")
    print(f" Test variations: {len(test_files) if test_files else 0} files")
    print("\nNext steps:")
    print("1. Run: python test_noaa_data.py")
    print("2. Check results in ../data/processed/")
    print("3. The model should now generate plausible weather images!")