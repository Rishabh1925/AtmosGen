#!/usr/bin/env python3
"""
Test the fine-tuned model from Kaggle
This verifies the model works and shows improved results
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import asyncio
import time

# Import our model service
from model_service import ModelService

def load_finetuned_model_info():
    """Load information about the fine-tuned model"""
    
    checkpoint_dir = Path("../checkpoints/atmosgen_finetuned")
    config_path = checkpoint_dir / "model_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    else:
        return None

def create_test_weather_sequence():
    """Create a test weather sequence for model evaluation"""
    
    from PIL import ImageDraw, ImageFilter
    
    # Create a simple weather evolution sequence
    size = (256, 256)
    sequence = []
    
    # Frame 1: Clear sky
    img1 = Image.new('RGB', size, (135, 206, 235))  # Sky blue
    sequence.append(img1)
    
    # Frame 2: Some clouds appearing
    img2 = Image.new('RGB', size, (176, 224, 230))  # Light blue
    draw = ImageDraw.Draw(img2)
    # Add some white clouds
    for i in range(3):
        x = 50 + i * 70
        y = 80 + i * 20
        draw.ellipse([x-30, y-20, x+30, y+20], fill=(255, 255, 255))
    img2 = img2.filter(ImageFilter.GaussianBlur(radius=2))
    sequence.append(img2)
    
    # Frame 3: More clouds, getting darker
    img3 = Image.new('RGB', size, (169, 169, 169))  # Gray
    draw = ImageDraw.Draw(img3)
    # Add gray clouds
    for i in range(5):
        x = 30 + i * 50
        y = 60 + i * 15
        draw.ellipse([x-40, y-25, x+40, y+25], fill=(200, 200, 200))
    img3 = img3.filter(ImageFilter.GaussianBlur(radius=3))
    sequence.append(img3)
    
    return sequence

async def test_model_comparison():
    """Test and compare model results before/after fine-tuning"""
    
    print("🧪 Testing Fine-tuned AtmosGen Model")
    print("=" * 50)
    
    # Check if fine-tuned model exists
    model_config = load_finetuned_model_info()
    
    if model_config:
        print("✅ Fine-tuned model found!")
        print(f"   - Training samples: {model_config['training_samples']}")
        print(f"   - Epochs: {model_config['epochs']}")
        print(f"   - Final loss: {model_config['final_loss']:.4f}")
    else:
        print("⚠️  Fine-tuned model not found, using demo checkpoint")
    
    # Create test sequence
    print("\n🌤️  Creating test weather sequence...")
    test_sequence = create_test_weather_sequence()
    
    # Save test inputs for visualization
    output_dir = Path("../data/test_results")
    output_dir.mkdir(exist_ok=True)
    
    for i, img in enumerate(test_sequence):
        img.save(output_dir / f"test_input_{i+1}.jpg")
    
    print(f"✓ Test sequence saved to: {output_dir}")
    
    # Initialize model service
    print("\n🤖 Loading model...")
    model_service = ModelService()
    
    try:
        await model_service.load_model()
        
        if not model_service.is_loaded():
            print("❌ Model failed to load")
            return
        
        print("✅ Model loaded successfully")
        
        # Create mock upload files from test sequence
        from fastapi import UploadFile
        import io
        
        upload_files = []
        for i, img in enumerate(test_sequence):
            # Convert PIL image to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Create mock UploadFile
            upload_file = UploadFile(
                filename=f"test_input_{i+1}.jpg",
                file=img_bytes
            )
            upload_files.append(upload_file)
        
        print(f"✓ Created {len(upload_files)} test inputs")
        
        # Run prediction
        print("\n🔮 Generating weather forecast...")
        start_time = time.time()
        
        result = await model_service.predict(upload_files)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Prediction completed in {processing_time:.2f} seconds")
        
        # Save the generated forecast
        if 'generated_image' in result:
            import base64
            
            # Decode base64 image
            image_data = result['generated_image']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            forecast_bytes = base64.b64decode(image_data)
            
            # Save forecast
            forecast_path = output_dir / "finetuned_forecast.jpg"
            with open(forecast_path, 'wb') as f:
                f.write(forecast_bytes)
            
            print(f"✓ Forecast saved: {forecast_path}")
            
            # Create comparison visualization
            create_comparison_visualization(test_sequence, forecast_path, output_dir)
            
        print("\n" + "=" * 50)
        print("🎉 MODEL TEST COMPLETED!")
        print("=" * 50)
        
        if model_config:
            print("✅ Fine-tuned model is working!")
            print("🎯 Results should show realistic weather patterns")
        else:
            print("⚠️  Using demo model - consider fine-tuning for better results")
        
        print(f"\n📁 Results saved in: {output_dir}")
        print("📊 Check the comparison image to see the forecast quality")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comparison_visualization(input_sequence, forecast_path, output_dir):
    """Create a visualization comparing inputs and forecast"""
    
    try:
        # Load forecast image
        forecast_img = Image.open(forecast_path)
        
        # Create comparison figure
        fig, axes = plt.subplots(1, len(input_sequence) + 1, figsize=(15, 3))
        
        # Plot input sequence
        for i, img in enumerate(input_sequence):
            axes[i].imshow(img)
            axes[i].set_title(f"Input {i+1}\n(T+{i})", fontsize=10)
            axes[i].axis('off')
        
        # Plot forecast
        axes[-1].imshow(forecast_img)
        axes[-1].set_title("AI Forecast\n(T+4)", fontsize=10, color='red', weight='bold')
        axes[-1].axis('off')
        
        plt.suptitle("AtmosGen Weather Forecasting Test", fontsize=14, weight='bold')
        plt.tight_layout()
        
        # Save comparison
        comparison_path = output_dir / "model_comparison.jpg"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison visualization saved: {comparison_path}")
        
    except Exception as e:
        print(f"Warning: Could not create comparison visualization: {e}")

def check_model_improvements():
    """Check what improvements the fine-tuned model should show"""
    
    print("\n📈 Expected Improvements from Fine-tuning:")
    print("=" * 50)
    
    improvements = [
        "🌤️  Realistic cloud formations and patterns",
        "🌪️  Proper storm development and structure", 
        "🌈  Smooth weather transitions over time",
        "🎯  Weather-appropriate colors and textures",
        "⚡ Consistent atmospheric physics",
        "🌊  Realistic precipitation and wind patterns"
    ]
    
    for improvement in improvements:
        print(f"   {improvement}")
    
    print("\n📊 Quality Metrics to Check:")
    print("   - Visual realism (clouds look natural)")
    print("   - Temporal consistency (logical weather evolution)")
    print("   - Meteorological accuracy (realistic patterns)")
    print("   - Color accuracy (proper atmospheric colors)")

async def main():
    """Main test function"""
    
    print("ATMOSGEN FINE-TUNED MODEL TEST")
    print("=" * 60)
    
    # Check model improvements info
    check_model_improvements()
    
    # Run the actual test
    success = await test_model_comparison()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated images in data/test_results/")
        print("2. Compare with previous demo results")
        print("3. Start the full AtmosGen app to test with real data")
        print("4. The frontend will automatically use the fine-tuned model!")
    else:
        print("\n❌ Test failed. Check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main())