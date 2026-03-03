#!/usr/bin/env python3
"""
Test script to process NOAA GOES-18 ABI .nc files and test with the AtmosGen model
"""

import os
import glob
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
import asyncio
from pathlib import Path
import base64
import io
import time

# Import our model service
from model_service import ModelService

def load_noaa_nc_file(filepath):
    """
    Load and process a NOAA GOES-18 ABI .nc file
    
    Args:
        filepath: Path to the .nc file
        
    Returns:
        numpy array of processed satellite data
    """
    try:
        # Open the NetCDF file
        ds = xr.open_dataset(filepath)
        
        print(f"Loading: {os.path.basename(filepath)}")
        print(f"Variables: {list(ds.variables.keys())}")
        print(f"Dimensions: {dict(ds.dims)}")
        
        # Get the radiance data (usually called 'Rad')
        if 'Rad' in ds.variables:
            radiance = ds['Rad'].values
        elif 'radiance' in ds.variables:
            radiance = ds['radiance'].values
        else:
            # Try to find the main data variable
            data_vars = [var for var in ds.variables if len(ds[var].dims) >= 2]
            if data_vars:
                radiance = ds[data_vars[0]].values
                print(f"Using variable: {data_vars[0]}")
            else:
                raise ValueError("Could not find radiance data in file")
        
        print(f"Radiance shape: {radiance.shape}")
        print(f"Radiance range: {np.nanmin(radiance):.3f} to {np.nanmax(radiance):.3f}")
        
        # Handle different data shapes
        if len(radiance.shape) == 3:
            # If 3D, take the first slice or average
            radiance = radiance[0] if radiance.shape[0] == 1 else np.mean(radiance, axis=0)
        
        # Remove NaN values and invalid data
        radiance = np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to 0-255 range for image processing
        if np.max(radiance) > np.min(radiance):
            radiance_norm = ((radiance - np.min(radiance)) / (np.max(radiance) - np.min(radiance)) * 255).astype(np.uint8)
        else:
            radiance_norm = np.zeros_like(radiance, dtype=np.uint8)
        
        ds.close()
        return radiance_norm
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def convert_to_image(data, output_path=None):
    """
    Convert numpy array to PIL Image
    
    Args:
        data: numpy array of satellite data
        output_path: optional path to save image
        
    Returns:
        PIL Image object
    """
    if data is None:
        return None
    
    # Ensure data is 2D
    if len(data.shape) > 2:
        data = data.squeeze()
    
    # Create PIL image
    image = Image.fromarray(data, mode='L')  # Grayscale
    
    # Convert to RGB for model compatibility
    image = image.convert('RGB')
    
    # Resize to model input size (assuming 256x256)
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    
    if output_path:
        image.save(output_path)
        print(f"Saved image: {output_path}")
    
    return image

def create_mock_upload_file(image, filename):
    """
    Create a mock UploadFile object from PIL Image for model testing
    """
    from fastapi import UploadFile
    import io
    
    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Create mock UploadFile with correct parameters
    upload_file = UploadFile(
        filename=filename,
        file=img_bytes
    )
    
    return upload_file

async def test_model_with_noaa_data():
    """
    Test the AtmosGen model with NOAA .nc files
    """
    print("=" * 60)
    print("TESTING ATMOSGEN MODEL WITH NOAA GOES-18 DATA")
    print("=" * 60)
    
    # Find all .nc files
    nc_files = sorted(glob.glob("../data/raw/*.nc"))
    
    if not nc_files:
        print("No .nc files found in ../data/raw/")
        return
    
    print(f"Found {len(nc_files)} NOAA .nc files")
    
    # Create output directory
    output_dir = Path("../data/processed")
    output_dir.mkdir(exist_ok=True)
    
    # Process first 4 files for a sequence
    sequence_length = min(4, len(nc_files))
    processed_images = []
    
    print(f"\nProcessing {sequence_length} files for model input...")
    
    for i, nc_file in enumerate(nc_files[:sequence_length]):
        print(f"\n--- Processing file {i+1}/{sequence_length} ---")
        
        # Load and process the .nc file
        radiance_data = load_noaa_nc_file(nc_file)
        
        if radiance_data is not None:
            # Convert to image
            filename = os.path.basename(nc_file).replace('.nc', '.jpg')
            output_path = output_dir / filename
            
            image = convert_to_image(radiance_data, output_path)
            
            if image:
                processed_images.append((image, filename))
                print(f" Successfully processed: {filename}")
            else:
                print(f" Failed to create image from: {filename}")
        else:
            print(f" Failed to load: {nc_file}")
    
    if len(processed_images) < 2:
        print(f"\n Need at least 2 images for model testing, got {len(processed_images)}")
        return
    
    print(f"\n Testing model with {len(processed_images)} processed images...")
    
    try:
        # Initialize model service
        model_service = ModelService()
        await model_service.load_model()
        
        if not model_service.is_loaded():
            print(" Model failed to load")
            return
        
        print(" Model loaded successfully")
        
        # Create mock upload files
        upload_files = []
        for image, filename in processed_images:
            upload_file = create_mock_upload_file(image, filename)
            upload_files.append(upload_file)
        
        print(f" Created {len(upload_files)} upload files")
        
        # Run prediction
        print("\n Running model prediction...")
        start_time = time.time()
        
        result = await model_service.predict(upload_files)
        
        processing_time = time.time() - start_time
        
        print(f" Prediction completed in {processing_time:.2f} seconds")
        print(f" Model processing time: {result['processing_time']:.2f} seconds")
        
        # Save the generated forecast
        if 'generated_image' in result:
            # Decode base64 image
            image_data = result['generated_image']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            forecast_bytes = base64.b64decode(image_data)
            
            # Save forecast
            forecast_path = output_dir / "noaa_forecast_result.jpg"
            with open(forecast_path, 'wb') as f:
                f.write(forecast_bytes)
            
            print(f" Forecast saved: {forecast_path}")
            
            # Also create a summary image showing input sequence + forecast
            create_summary_image(processed_images, forecast_path, output_dir / "noaa_test_summary.jpg")
            
        print("\n" + "=" * 60)
        print(" MODEL TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Input files: {len(processed_images)} NOAA GOES-18 images")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Output: Weather forecast generated")
        print(f"Results saved in: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_summary_image(input_images, forecast_path, output_path):
    """
    Create a summary image showing input sequence and forecast result
    """
    try:
        # Load forecast image
        forecast_img = Image.open(forecast_path)
        
        # Create figure
        fig, axes = plt.subplots(1, len(input_images) + 1, figsize=(15, 3))
        
        # Plot input images
        for i, (img, filename) in enumerate(input_images):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Input {i+1}\n{filename[:20]}...", fontsize=8)
            axes[i].axis('off')
        
        # Plot forecast
        axes[-1].imshow(forecast_img)
        axes[-1].set_title("AI Forecast\n(Generated)", fontsize=8, color='red')
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Summary image saved: {output_path}")
        
    except Exception as e:
        print(f"Warning: Could not create summary image: {e}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_model_with_noaa_data())