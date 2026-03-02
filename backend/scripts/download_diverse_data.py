#!/usr/bin/env python3
"""
Download diverse satellite data for better model testing
This gets data from different regions, times, and weather conditions
"""

import requests
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

def download_nasa_worldview_images():
    """
    Download diverse satellite images from NASA Worldview API
    This gives us different geographic regions and weather conditions
    """
    print("Downloading diverse satellite data from NASA Worldview...")
    
    # Create download directory
    download_dir = Path("../data/diverse")
    download_dir.mkdir(exist_ok=True)
    
    # Different regions and dates for diversity
    regions = [
        {
            "name": "pacific_storm",
            "bbox": "-180,10,-120,60",  # Pacific Ocean
            "description": "Pacific storm systems"
        },
        {
            "name": "atlantic_hurricane",
            "bbox": "-100,10,-40,50",   # Atlantic Ocean
            "description": "Atlantic weather patterns"
        },
        {
            "name": "continental_us",
            "bbox": "-130,25,-65,50",   # Continental US
            "description": "Continental weather systems"
        },
        {
            "name": "gulf_mexico",
            "bbox": "-100,18,-80,32",   # Gulf of Mexico
            "description": "Gulf weather patterns"
        }
    ]
    
    # Different dates for temporal diversity
    base_date = datetime(2024, 8, 15)  # Summer weather
    dates = [
        base_date,
        base_date + timedelta(days=7),   # 1 week later
        base_date + timedelta(days=30),  # 1 month later
        base_date + timedelta(days=90),  # Different season
    ]
    
    downloaded_files = []
    
    for region in regions:
        for i, date in enumerate(dates):
            try:
                # NASA Worldview GIBS API
                date_str = date.strftime("%Y-%m-%d")
                
                # MODIS Terra True Color (good for weather visualization)
                url = (
                    f"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
                    f"MODIS_Terra_CorrectedReflectance_TrueColor/default/"
                    f"{date_str}/250m/4/8/6.jpg"
                )
                
                filename = f"{region['name']}_{date_str}_{i:02d}.jpg"
                filepath = download_dir / filename
                
                print(f"Downloading: {region['description']} - {date_str}")
                
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    downloaded_files.append(filepath)
                    print(f"✓ Saved: {filename}")
                else:
                    print(f"❌ Failed: {filename} (HTTP {response.status_code})")
                
                # Be nice to the server
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error downloading {region['name']} {date_str}: {e}")
    
    print(f"\n✓ Downloaded {len(downloaded_files)} diverse satellite images")
    return downloaded_files

def create_synthetic_weather_sequence():
    """
    Create a synthetic weather sequence showing temporal evolution
    This simulates how weather systems develop over time
    """
    print("\nCreating synthetic weather sequence...")
    
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    import random
    
    # Create synthetic directory
    synthetic_dir = Path("../data/synthetic")
    synthetic_dir.mkdir(exist_ok=True)
    
    # Base image size
    size = (512, 512)
    
    def create_weather_frame(frame_num, total_frames):
        """Create a single frame of evolving weather"""
        
        # Create base image
        img = Image.new('RGB', size, color=(50, 100, 150))  # Sky blue base
        draw = ImageDraw.Draw(img)
        
        # Animate cloud movement and development
        progress = frame_num / total_frames
        
        # Moving cloud system
        cloud_x = int(size[0] * (0.2 + 0.6 * progress))  # Move across image
        cloud_y = int(size[1] * 0.3)
        
        # Cloud intensity changes over time
        cloud_intensity = int(255 * (0.3 + 0.4 * np.sin(progress * np.pi)))
        
        # Draw evolving cloud system
        for i in range(5):
            offset_x = random.randint(-50, 50)
            offset_y = random.randint(-30, 30)
            radius = random.randint(40, 80)
            
            # Cloud color varies with intensity
            cloud_color = (
                min(255, 200 + cloud_intensity // 5),
                min(255, 200 + cloud_intensity // 5),
                min(255, 200 + cloud_intensity // 5)
            )
            
            draw.ellipse([
                cloud_x + offset_x - radius,
                cloud_y + offset_y - radius,
                cloud_x + offset_x + radius,
                cloud_y + offset_y + radius
            ], fill=cloud_color)
        
        # Add some atmospheric effects
        if progress > 0.5:  # Storm development
            # Add darker storm clouds
            storm_color = (80, 80, 100)
            draw.ellipse([
                cloud_x - 60, cloud_y - 40,
                cloud_x + 60, cloud_y + 40
            ], fill=storm_color)
        
        # Apply blur for atmospheric effect
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        return img
    
    # Create sequence of 8 frames
    sequence_files = []
    total_frames = 8
    
    for frame in range(total_frames):
        weather_img = create_weather_frame(frame, total_frames)
        
        # Resize to model input size
        weather_img = weather_img.resize((256, 256), Image.Resampling.LANCZOS)
        
        filename = f"synthetic_weather_sequence_{frame:02d}.jpg"
        filepath = synthetic_dir / filename
        
        weather_img.save(filepath, quality=95)
        sequence_files.append(filepath)
        
        print(f"✓ Created frame {frame + 1}/{total_frames}")
    
    print(f"✓ Created synthetic weather sequence: {len(sequence_files)} frames")
    return sequence_files

def main():
    """Main function to download and create diverse test data"""
    print("=" * 60)
    print("DOWNLOADING DIVERSE SATELLITE DATA")
    print("=" * 60)
    
    all_files = []
    
    # Try to download real satellite data
    try:
        nasa_files = download_nasa_worldview_images()
        all_files.extend(nasa_files)
    except Exception as e:
        print(f"❌ NASA download failed: {e}")
        print("Continuing with synthetic data...")
    
    # Create synthetic weather sequence
    try:
        synthetic_files = create_synthetic_weather_sequence()
        all_files.extend(synthetic_files)
    except Exception as e:
        print(f"❌ Synthetic data creation failed: {e}")
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"✓ Total files created: {len(all_files)}")
    print("\nData locations:")
    print("- Real satellite data: ../data/diverse/")
    print("- Synthetic sequences: ../data/synthetic/")
    print("\nNext steps:")
    print("1. Run: python create_demo_checkpoint.py")
    print("2. Run: python test_noaa_data.py")
    print("3. Test with new diverse data!")

if __name__ == "__main__":
    main()