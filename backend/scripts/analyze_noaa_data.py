#!/usr/bin/env python3
"""
Analyze NOAA data to understand why images look similar and check data quality
"""

import os
import glob
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_nc_file(filepath):
    """Analyze a single .nc file in detail"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        ds = xr.open_dataset(filepath)
        
        # Extract timestamp from filename
        filename = os.path.basename(filepath)
        # Format: OR_ABI-L1b-RadC-M6C13_G18_s20260361201174_e20260361203560_c20260361204039.nc
        # s20260361201174 = start time: 2026 day 036 (Feb 5) 20:11:74 (20:11:17.4)
        start_time = filename.split('_s')[1].split('_')[0]
        year = int(start_time[:4])
        day_of_year = int(start_time[4:7])
        hour_min_sec = start_time[7:]
        
        print(f"Year: {year}")
        print(f"Day of year: {day_of_year}")
        print(f"Time: {hour_min_sec}")
        
        # Get radiance data
        radiance = ds['Rad'].values
        print(f"Radiance shape: {radiance.shape}")
        print(f"Radiance dtype: {radiance.dtype}")
        print(f"Radiance range: {np.nanmin(radiance):.6f} to {np.nanmax(radiance):.6f}")
        print(f"Radiance mean: {np.nanmean(radiance):.6f}")
        print(f"Radiance std: {np.nanstd(radiance):.6f}")
        
        # Check for NaN values
        nan_count = np.sum(np.isnan(radiance))
        total_pixels = radiance.size
        print(f"NaN pixels: {nan_count}/{total_pixels} ({100*nan_count/total_pixels:.2f}%)")
        
        # Check data quality flags
        if 'DQF' in ds.variables:
            dqf = ds['DQF'].values
            print(f"Data Quality Flags shape: {dqf.shape}")
            print(f"DQF unique values: {np.unique(dqf)}")
        
        # Get geographic info
        if 'geospatial_lat_lon_extent' in ds.variables:
            extent = ds['geospatial_lat_lon_extent']
            print(f"Geographic extent: {extent}")
        
        # Sample a small region to see actual values
        center_y, center_x = radiance.shape[0]//2, radiance.shape[1]//2
        sample_region = radiance[center_y-5:center_y+5, center_x-5:center_x+5]
        print(f"Sample 10x10 region at center:")
        print(sample_region)
        
        ds.close()
        return {
            'filename': filename,
            'radiance_min': np.nanmin(radiance),
            'radiance_max': np.nanmax(radiance),
            'radiance_mean': np.nanmean(radiance),
            'radiance_std': np.nanstd(radiance),
            'nan_percent': 100*nan_count/total_pixels,
            'sample_region': sample_region
        }
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

def compare_processed_images():
    """Compare the processed images to see if they're actually different"""
    print(f"\n{'='*60}")
    print("COMPARING PROCESSED IMAGES")
    print(f"{'='*60}")
    
    processed_files = sorted(glob.glob("../data/processed/OR_ABI*.jpg"))
    
    if len(processed_files) < 2:
        print("Not enough processed images to compare")
        return
    
    images = []
    for filepath in processed_files:
        img = Image.open(filepath)
        img_array = np.array(img)
        images.append(img_array)
        print(f"Loaded {os.path.basename(filepath)}: shape {img_array.shape}")
    
    # Compare images pairwise
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            img1, img2 = images[i], images[j]
            
            # Calculate differences
            diff = np.abs(img1.astype(float) - img2.astype(float))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            # Calculate correlation
            correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
            
            print(f"\nImage {i+1} vs Image {j+1}:")
            print(f"  Max difference: {max_diff:.2f}")
            print(f"  Mean difference: {mean_diff:.2f}")
            print(f"  Correlation: {correlation:.6f}")
            
            if correlation > 0.99:
                print(f"    Images are nearly identical!")
            elif correlation > 0.95:
                print(f"    Images are very similar")
            else:
                print(f"   Images are different")

def create_detailed_visualization():
    """Create a detailed visualization of the raw data"""
    print(f"\n{'='*60}")
    print("CREATING DETAILED VISUALIZATION")
    print(f"{'='*60}")
    
    nc_files = sorted(glob.glob("../data/raw/*.nc"))[:4]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, nc_file in enumerate(nc_files):
        try:
            ds = xr.open_dataset(nc_file)
            radiance = ds['Rad'].values
            
            # Raw radiance (top row)
            im1 = axes[0, i].imshow(radiance, cmap='gray', vmin=np.nanpercentile(radiance, 1), vmax=np.nanpercentile(radiance, 99))
            axes[0, i].set_title(f'Raw Radiance {i+1}\nRange: {np.nanmin(radiance):.2f}-{np.nanmax(radiance):.2f}')
            axes[0, i].axis('off')
            
            # Processed version (bottom row)
            # Normalize like in the processing script
            radiance_clean = np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(radiance_clean) > np.min(radiance_clean):
                radiance_norm = ((radiance_clean - np.min(radiance_clean)) / 
                               (np.max(radiance_clean) - np.min(radiance_clean)) * 255).astype(np.uint8)
            else:
                radiance_norm = np.zeros_like(radiance_clean, dtype=np.uint8)
            
            axes[1, i].imshow(radiance_norm, cmap='gray')
            axes[1, i].set_title(f'Processed {i+1}\nNormalized 0-255')
            axes[1, i].axis('off')
            
            ds.close()
            
        except Exception as e:
            print(f"Error processing {nc_file}: {e}")
    
    plt.tight_layout()
    plt.savefig('../data/processed/detailed_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(" Detailed visualization saved: ../data/processed/detailed_analysis.png")

def main():
    """Main analysis function"""
    print("NOAA GOES-18 DATA ANALYSIS")
    print("="*60)
    
    # Find all .nc files
    nc_files = sorted(glob.glob("../data/raw/*.nc"))
    
    if not nc_files:
        print("No .nc files found!")
        return
    
    print(f"Found {len(nc_files)} NOAA .nc files")
    
    # Analyze each file
    analysis_results = []
    for nc_file in nc_files[:4]:  # Analyze first 4
        result = analyze_nc_file(nc_file)
        if result:
            analysis_results.append(result)
    
    # Summary comparison
    if len(analysis_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        for i, result in enumerate(analysis_results):
            print(f"File {i+1}: {result['filename'][:30]}...")
            print(f"  Mean radiance: {result['radiance_mean']:.6f}")
            print(f"  Std radiance: {result['radiance_std']:.6f}")
            print(f"  Range: {result['radiance_min']:.6f} to {result['radiance_max']:.6f}")
        
        # Check if all files have very similar statistics
        means = [r['radiance_mean'] for r in analysis_results]
        stds = [r['radiance_std'] for r in analysis_results]
        
        mean_variation = (max(means) - min(means)) / np.mean(means)
        std_variation = (max(stds) - min(stds)) / np.mean(stds)
        
        print(f"\nVariation between files:")
        print(f"  Mean variation: {mean_variation*100:.2f}%")
        print(f"  Std variation: {std_variation*100:.2f}%")
        
        if mean_variation < 0.01 and std_variation < 0.01:
            print("  WARNING: Files have very similar statistics - might be the same scene!")
        else:
            print(" Files show reasonable variation")
    
    # Compare processed images
    compare_processed_images()
    
    # Create detailed visualization
    create_detailed_visualization()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("\nKey Findings:")
    print("1. Check if input images are actually different")
    print("2. Model is using random weights (no training)")
    print("3. Results saved in ../data/processed/")

if __name__ == "__main__":
    main()