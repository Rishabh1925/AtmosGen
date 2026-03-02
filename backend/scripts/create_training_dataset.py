#!/usr/bin/env python3
"""
Create synthetic weather training dataset for fine-tuning
This generates realistic weather sequences for training the model
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
from pathlib import Path
import json
from datetime import datetime, timedelta

def create_weather_base_images():
    """Create base weather patterns"""
    
    patterns = {
        'clear_sky': {
            'base_color': (135, 206, 235),  # Sky blue
            'cloud_coverage': 0.1,
            'description': 'Clear sunny day'
        },
        'partly_cloudy': {
            'base_color': (176, 224, 230),  # Light blue
            'cloud_coverage': 0.3,
            'description': 'Partly cloudy'
        },
        'overcast': {
            'base_color': (169, 169, 169),  # Gray
            'cloud_coverage': 0.8,
            'description': 'Overcast sky'
        },
        'stormy': {
            'base_color': (105, 105, 105),  # Dark gray
            'cloud_coverage': 0.9,
            'description': 'Storm system'
        },
        'hurricane': {
            'base_color': (70, 70, 70),     # Very dark
            'cloud_coverage': 1.0,
            'description': 'Hurricane/cyclone'
        }
    }
    
    return patterns

def generate_cloud_layer(size, coverage, intensity, pattern_type='normal'):
    """Generate a cloud layer with specified coverage and intensity"""
    
    width, height = size
    cloud_layer = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(cloud_layer)
    
    # Number of cloud formations based on coverage
    num_clouds = int(coverage * 20)
    
    for _ in range(num_clouds):
        # Random cloud position and size
        x = random.randint(-50, width + 50)
        y = random.randint(-50, height + 50)
        
        if pattern_type == 'hurricane':
            # Spiral pattern for hurricanes
            center_x, center_y = width // 2, height // 2
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(50, min(width, height) // 3)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
        
        # Cloud size varies with pattern type
        if pattern_type == 'hurricane':
            cloud_size = random.randint(30, 80)
        else:
            cloud_size = random.randint(20, 60)
        
        # Cloud opacity based on intensity
        opacity = int(255 * intensity * random.uniform(0.5, 1.0))
        
        # Cloud color (white to gray)
        cloud_color = (255, 255, 255, opacity)
        if pattern_type == 'stormy':
            # Darker clouds for storms
            gray_level = random.randint(150, 200)
            cloud_color = (gray_level, gray_level, gray_level, opacity)
        
        # Draw cloud as ellipse
        draw.ellipse([
            x - cloud_size, y - cloud_size,
            x + cloud_size, y + cloud_size
        ], fill=cloud_color)
    
    # Apply blur for realistic cloud effect
    blur_radius = 2 if pattern_type == 'hurricane' else 3
    cloud_layer = cloud_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return cloud_layer

def create_weather_image(pattern_name, pattern_info, size=(512, 512), variation=0):
    """Create a single weather image with specified pattern"""
    
    # Base sky color
    base_img = Image.new('RGB', size, pattern_info['base_color'])
    
    # Add some texture to the base
    if variation > 0:
        # Slight color variations
        enhancer = ImageEnhance.Color(base_img)
        base_img = enhancer.enhance(0.8 + 0.4 * variation)
        
        enhancer = ImageEnhance.Brightness(base_img)
        base_img = enhancer.enhance(0.9 + 0.2 * variation)
    
    # Generate cloud layers
    cloud_coverage = pattern_info['cloud_coverage']
    
    if cloud_coverage > 0:
        # Main cloud layer
        clouds = generate_cloud_layer(
            size, 
            cloud_coverage, 
            intensity=0.7,
            pattern_type=pattern_name
        )
        
        # Composite clouds onto base
        base_img = Image.alpha_composite(
            base_img.convert('RGBA'), 
            clouds
        ).convert('RGB')
        
        # Add secondary cloud layer for depth
        if cloud_coverage > 0.5:
            upper_clouds = generate_cloud_layer(
                size,
                cloud_coverage * 0.3,
                intensity=0.4,
                pattern_type=pattern_name
            )
            
            base_img = Image.alpha_composite(
                base_img.convert('RGBA'),
                upper_clouds
            ).convert('RGB')
    
    # Add atmospheric effects
    if pattern_name == 'stormy':
        # Darken for storm
        enhancer = ImageEnhance.Brightness(base_img)
        base_img = enhancer.enhance(0.7)
    elif pattern_name == 'hurricane':
        # Add spiral distortion effect
        base_img = add_hurricane_spiral(base_img)
    
    return base_img

def add_hurricane_spiral(image):
    """Add spiral distortion to simulate hurricane structure"""
    
    # This is a simplified spiral effect
    # In a real implementation, you'd use more sophisticated image warping
    
    # For now, just add some radial blur and darkening
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(0.6)
    
    # Add some radial structure with drawing
    draw = ImageDraw.Draw(image)
    center_x, center_y = image.size[0] // 2, image.size[1] // 2
    
    # Draw spiral arms (simplified)
    for i in range(3):
        angle_offset = i * (2 * np.pi / 3)
        for r in range(20, min(image.size) // 2, 10):
            angle = angle_offset + r * 0.1
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            
            # Draw small cloud formations along spiral
            draw.ellipse([x-5, y-5, x+5, y+5], fill=(200, 200, 200))
    
    return image

def create_weather_sequence(pattern_sequence, sequence_length=4):
    """Create a temporal sequence showing weather evolution"""
    
    patterns = create_weather_base_images()
    sequence = []
    
    for i, pattern_name in enumerate(pattern_sequence):
        if pattern_name not in patterns:
            pattern_name = 'partly_cloudy'  # Default fallback
        
        pattern_info = patterns[pattern_name]
        
        # Create variation based on position in sequence
        variation = i / (len(pattern_sequence) - 1) if len(pattern_sequence) > 1 else 0
        
        weather_img = create_weather_image(pattern_name, pattern_info, variation=variation)
        
        # Resize to model input size
        weather_img = weather_img.resize((256, 256), Image.Resampling.LANCZOS)
        
        sequence.append(weather_img)
    
    return sequence

def generate_training_sequences():
    """Generate diverse weather sequences for training"""
    
    # Define weather evolution patterns
    evolution_patterns = [
        # Clear to cloudy
        ['clear_sky', 'partly_cloudy', 'overcast', 'overcast'],
        
        # Storm development
        ['partly_cloudy', 'overcast', 'stormy', 'stormy'],
        
        # Storm clearing
        ['stormy', 'overcast', 'partly_cloudy', 'clear_sky'],
        
        # Hurricane formation
        ['overcast', 'stormy', 'hurricane', 'hurricane'],
        
        # Stable clear weather
        ['clear_sky', 'clear_sky', 'partly_cloudy', 'partly_cloudy'],
        
        # Stable cloudy weather
        ['overcast', 'overcast', 'overcast', 'partly_cloudy'],
        
        # Variable conditions
        ['partly_cloudy', 'overcast', 'partly_cloudy', 'clear_sky'],
        
        # Storm passing
        ['clear_sky', 'stormy', 'overcast', 'partly_cloudy'],
    ]
    
    training_data = []
    
    for pattern_idx, pattern in enumerate(evolution_patterns):
        # Create multiple variations of each pattern
        for variation in range(5):  # 5 variations per pattern
            
            # Generate sequence
            sequence = create_weather_sequence(pattern)
            
            # Input: first 3 images, Target: last image (forecast)
            input_sequence = sequence[:3]
            target_forecast = sequence[3]
            
            # Create sample metadata
            sample_data = {
                'pattern': pattern,
                'variation': variation,
                'input_files': [],
                'target_file': '',
                'description': f"Weather evolution: {' → '.join(pattern)}"
            }
            
            training_data.append({
                'input_sequence': input_sequence,
                'target_forecast': target_forecast,
                'metadata': sample_data
            })
    
    return training_data

def save_training_dataset(training_data, output_dir):
    """Save the training dataset to disk"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_path / 'inputs').mkdir(exist_ok=True)
    (output_path / 'targets').mkdir(exist_ok=True)
    
    dataset_info = {
        'total_samples': len(training_data),
        'created_at': datetime.now().isoformat(),
        'description': 'Synthetic weather sequences for AtmosGen training',
        'samples': []
    }
    
    for idx, sample in enumerate(training_data):
        sample_id = f"sample_{idx:04d}"
        
        # Save input sequence
        input_files = []
        for seq_idx, img in enumerate(sample['input_sequence']):
            input_filename = f"{sample_id}_input_{seq_idx}.jpg"
            input_path = output_path / 'inputs' / input_filename
            img.save(input_path, quality=95)
            input_files.append(input_filename)
        
        # Save target forecast
        target_filename = f"{sample_id}_target.jpg"
        target_path = output_path / 'targets' / target_filename
        sample['target_forecast'].save(target_path, quality=95)
        
        # Update metadata
        sample_info = sample['metadata'].copy()
        sample_info['sample_id'] = sample_id
        sample_info['input_files'] = input_files
        sample_info['target_file'] = target_filename
        
        dataset_info['samples'].append(sample_info)
        
        if (idx + 1) % 10 == 0:
            print(f"✓ Saved {idx + 1}/{len(training_data)} samples")
    
    # Save dataset metadata
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✅ Training dataset saved to: {output_path}")
    print(f"📊 Total samples: {len(training_data)}")
    print(f"📁 Input images: {len(training_data) * 3}")
    print(f"🎯 Target images: {len(training_data)}")
    
    return output_path

def main():
    """Main function to create the training dataset"""
    
    print("=" * 60)
    print("CREATING ATMOSGEN TRAINING DATASET")
    print("=" * 60)
    
    print("🌤️  Generating weather sequences...")
    training_data = generate_training_sequences()
    
    print(f"✓ Generated {len(training_data)} training sequences")
    
    print("\n💾 Saving training dataset...")
    output_dir = "../data/training"
    dataset_path = save_training_dataset(training_data, output_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING DATASET CREATED!")
    print("=" * 60)
    print(f"📍 Location: {dataset_path}")
    print(f"📊 Ready for Kaggle upload")
    print("\nNext steps:")
    print("1. Run: python package_for_kaggle.py")
    print("2. Upload to Kaggle")
    print("3. Follow the training guide!")

if __name__ == "__main__":
    main()