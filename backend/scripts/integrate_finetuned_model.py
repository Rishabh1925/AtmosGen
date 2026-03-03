#!/usr/bin/env python3
"""
Integrate fine-tuned model from Kaggle into AtmosGen
Run this after downloading the checkpoint from Kaggle
"""

import os
import zipfile
import shutil
from pathlib import Path
import json

def integrate_finetuned_model():
    """Integrate the downloaded fine-tuned model"""
    
    print(" Integrating fine-tuned model into AtmosGen...")
    
    # Look for downloaded checkpoint
    checkpoint_zip = Path("atmosgen_checkpoint.zip")
    
    if not checkpoint_zip.exists():
        print(" Checkpoint not found!")
        print("Please download 'atmosgen_checkpoint.zip' from Kaggle and place it in this directory.")
        return False
    
    # Extract checkpoint
    print(" Extracting checkpoint...")
    checkpoint_dir = Path("../checkpoints/atmosgen_finetuned")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(checkpoint_zip, 'r') as zip_ref:
        zip_ref.extractall(checkpoint_dir)
    
    # Verify extraction
    required_files = ['unet', 'text_encoder', 'tokenizer', 'scheduler', 'model_config.json']
    
    for required_file in required_files:
        if not (checkpoint_dir / required_file).exists():
            print(f" Missing required file: {required_file}")
            return False
    
    print(" Checkpoint extracted successfully!")
    
    # Load model config
    with open(checkpoint_dir / 'model_config.json', 'r') as f:
        config = json.load(f)
    
    print(f" Model info:")
    print(f"   - Type: {config['model_type']}")
    print(f"   - Base model: {config['base_model']}")
    print(f"   - Training samples: {config['training_samples']}")
    print(f"   - Epochs: {config['epochs']}")
    print(f"   - Final loss: {config['final_loss']:.4f}")
    
    # Update model service to use fine-tuned model
    print(" Updating model service...")
    
    # The model service will automatically detect and use the fine-tuned model
    # because it looks for the most recent checkpoint
    
    print(" Integration complete!")
    print("
Next steps:")
    print("1. Test the model: python test_finetuned_model.py")
    print("2. Start the backend: python main.py")
    print("3. The frontend will now use the fine-tuned model!")
    
    return True

if __name__ == "__main__":
    integrate_finetuned_model()
