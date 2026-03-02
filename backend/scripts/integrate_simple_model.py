#!/usr/bin/env python3
"""
Integration script for the simple weather forecasting model trained on Kaggle.
This replaces the complex diffusion model with a simple, reliable CNN.
"""

import os
import sys
import zipfile
import json
import torch
import torch.nn as nn
from pathlib import Path

def create_simple_model():
    """Create the same SimpleWeatherNet architecture used in training."""
    
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

def integrate_simple_model():
    """Integrate the trained simple weather model."""
    
    print(" Integrating Simple Weather Model...")
    
    # Check if the model zip exists
    model_zip_path = "../atmosgen_simple_model.zip"
    if not os.path.exists(model_zip_path):
        print(" Error: atmosgen_simple_model.zip not found!")
        print("📥 Please download it from Kaggle and place it in the project root directory")
        return False
    
    # Extract the model
    extract_dir = "simple_model"
    if os.path.exists(extract_dir):
        import shutil
        shutil.rmtree(extract_dir)
    
    print(" Extracting model...")
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Load model config
    config_path = os.path.join(extract_dir, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f" Model: {config['model_name']}")
        print(f" Training samples: {config['training_samples']}")
        print(f" Final loss: {config['final_loss']:.4f}")
    
    # Create model architecture
    print(" Creating model architecture...")
    model = create_simple_model()
    
    # Load trained weights
    model_path = os.path.join(extract_dir, "weather_model.pth")
    if os.path.exists(model_path):
        print("📥 Loading trained weights...")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(" Model weights loaded successfully!")
    else:
        print(" Error: weather_model.pth not found in the extracted files")
        return False
    
    # Save to checkpoints directory
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "atmosgen_simple_v1.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config if 'config' in locals() else {},
        'model_type': 'SimpleWeatherNet'
    }, checkpoint_path)
    
    print(f" Model saved to: {checkpoint_path}")
    
    # Update model service configuration
    print(" Updating model service...")
    
    # Create a simple model service update
    service_update = '''
# Simple Weather Model Integration
# Add this to your model_service.py

class SimpleWeatherService:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load the simple weather model."""
        checkpoint_path = "../checkpoints/atmosgen_simple_v1.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Create model architecture
            from integrate_simple_model import create_simple_model
            self.model = create_simple_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(" Simple weather model loaded!")
        else:
            print(" Simple weather model not found!")
    
    def predict_weather(self, input_image):
        """Generate weather prediction from input image."""
        if self.model is None:
            return None
        
        with torch.no_grad():
            # Preprocess input
            if isinstance(input_image, str):
                from PIL import Image
                import torchvision.transforms as transforms
                
                image = Image.open(input_image).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_tensor = transform(image).unsqueeze(0).to(self.device)
            else:
                input_tensor = input_image.to(self.device)
            
            # Generate prediction
            prediction = self.model(input_tensor)
            
            return prediction
'''
    
    with open("simple_model_service.py", 'w') as f:
        f.write(service_update)
    
    print(" Simple weather model integration complete!")
    print("\n Next steps:")
    print("1. Update your model_service.py to use SimpleWeatherService")
    print("2. Test the model with: python test_simple_model.py")
    print("3. Start your application: python main.py")
    
    return True

if __name__ == "__main__":
    success = integrate_simple_model()
    if success:
        print("\n Integration successful!")
    else:
        print("\n Integration failed!")
        sys.exit(1)