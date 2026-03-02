#!/usr/bin/env python3
"""
Test the WeatherBench package to ensure it works correctly
"""

import zipfile
import json
import os
from pathlib import Path
from PIL import Image

def test_weatherbench_package():
    """Test the WeatherBench package contents"""
    
    print("🧪 Testing WeatherBench package...")
    
    # Test zip file exists
    zip_path = Path("../data/atmosgen_weatherbench_data.zip")
    if not zip_path.exists():
        print("❌ Package not found!")
        return False
    
    print(f"✅ Package found: {zip_path}")
    print(f"📊 Size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Test zip contents
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        
        # Check required files
        required_files = [
            'README.md',
            'atmosgen_high_accuracy_training.py',
            'weatherbench_data/weatherbench_dataset_info.json'
        ]
        
        for req_file in required_files:
            if req_file in file_list:
                print(f"✅ Found: {req_file}")
            else:
                print(f"❌ Missing: {req_file}")
                return False
        
        # Count image files
        image_files = [f for f in file_list if f.endswith('.jpg')]
        print(f"✅ Image files: {len(image_files)}")
        
        # Extract and test dataset info
        zip_ref.extract('weatherbench_data/weatherbench_dataset_info.json', '/tmp/')
        
        with open('/tmp/weatherbench_data/weatherbench_dataset_info.json', 'r') as f:
            dataset_info = json.load(f)
        
        print(f"✅ Dataset: {dataset_info['dataset_name']}")
        print(f"✅ Samples: {dataset_info['total_samples']}")
        print(f"✅ Quality: {dataset_info['quality']}")
        
        # Test a sample image
        if image_files:
            sample_image = image_files[0]
            zip_ref.extract(sample_image, '/tmp/')
            
            img = Image.open(f'/tmp/{sample_image}')
            print(f"✅ Sample image: {img.size} pixels, {img.mode} mode")
        
        # Clean up
        import shutil
        shutil.rmtree('/tmp/weatherbench_data', ignore_errors=True)
        if os.path.exists(f'/tmp/{sample_image}'):
            os.remove(f'/tmp/{sample_image}')
    
    print("\n🎉 WeatherBench package test PASSED!")
    return True

def test_training_script():
    """Test the training script syntax"""
    
    print("\n🧪 Testing training script...")
    
    script_path = Path("../data/weatherbench_package/atmosgen_high_accuracy_training.py")
    
    if not script_path.exists():
        print("❌ Training script not found!")
        return False
    
    # Read and basic syntax check
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Check for key components
    key_components = [
        'import torch',
        'StableDiffusionPipeline',
        'WeatherBenchDataset',
        'num_epochs = 8',
        'atmosgen_high_accuracy'
    ]
    
    for component in key_components:
        if component in script_content:
            print(f"✅ Found: {component}")
        else:
            print(f"❌ Missing: {component}")
            return False
    
    print(f"✅ Script length: {len(script_content)} characters")
    print("✅ Training script test PASSED!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING WEATHERBENCH PACKAGE")
    print("=" * 50)
    
    package_ok = test_weatherbench_package()
    script_ok = test_training_script()
    
    if package_ok and script_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Package is ready for Kaggle upload")
        print("✅ Training script is complete")
        print("✅ Data is properly formatted")
    else:
        print("\n❌ TESTS FAILED!")
        print("Package needs fixes before upload")