#!/usr/bin/env python3
"""
Simple test script for AtmosGen API
"""

import requests
import json
import os
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_predict():
    """Test prediction endpoint with sample images"""
    print("\n🔍 Testing prediction endpoint...")
    
    # Look for sample images
    image_dir = Path("core_model/data/satellite_images")
    if not image_dir.exists():
        print("❌ No sample images found")
        return False
    
    image_files = list(image_dir.glob("*.png"))[:3]  # Take first 3 images
    
    if len(image_files) == 0:
        print("❌ No PNG images found")
        return False
    
    print(f"📸 Using {len(image_files)} sample images")
    
    try:
        files = []
        for img_path in image_files:
            files.append(('files', (img_path.name, open(img_path, 'rb'), 'image/png')))
        
        response = requests.post(f"{API_BASE_URL}/predict", files=files)
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['message']}")
            print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
            print(f"📊 Generated image size: {len(result['generated_image'])} chars")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def main():
    print("🧪 AtmosGen API Test Suite")
    print("=" * 40)
    
    # Test health
    health_ok = test_health()
    
    if not health_ok:
        print("\n❌ Backend is not running or not healthy")
        print("💡 Start the backend with: python backend/main.py")
        return
    
    # Test prediction
    predict_ok = test_predict()
    
    print("\n" + "=" * 40)
    if health_ok and predict_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()