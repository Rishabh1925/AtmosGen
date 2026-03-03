#!/usr/bin/env python3
"""
Process GOES-18 Band 13 (10.3μm IR) NetCDF files into training data.

Converts radiance to brightness temperature, creates image tiles,
and generates cloud masks using Otsu thresholding.

Usage:
    python backend/scripts/process_goes_data.py
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def radiance_to_brightness_temp(radiance, planck_fk1, planck_fk2, planck_bc1, planck_bc2):
    """
    Convert spectral radiance to brightness temperature using GOES-18 Planck coefficients.
    These coefficients are stored in the NetCDF file attributes.
    """
    # Effective temperature
    T_eff = (planck_fk2 / np.log((planck_fk1 / radiance) + 1.0))
    # Actual brightness temperature
    T_b = (T_eff - planck_bc1) / planck_bc2
    return T_b


def normalize_brightness_temp(bt, vmin=200.0, vmax=310.0):
    """
    Normalize brightness temperature to [0, 1] range.
    - vmin ~ 200K: very cold cloud tops
    - vmax ~ 310K: warm ground surface
    """
    bt_clipped = np.clip(bt, vmin, vmax)
    return (bt_clipped - vmin) / (vmax - vmin)


def create_cloud_mask(bt_normalized, threshold=None):
    """
    Create binary cloud mask from normalized brightness temperature.
    
    In IR imagery:
    - Low brightness temp (dark/low values) = cold = cloud tops
    - High brightness temp (bright/high values) = warm = ground
    
    If threshold is None, uses Otsu's method.
    """
    from skimage.filters import threshold_otsu
    
    if threshold is None:
        try:
            threshold = threshold_otsu(bt_normalized[~np.isnan(bt_normalized)])
        except ValueError:
            threshold = 0.5
    
    # Clouds are COLD (low brightness temp = low normalized value)
    cloud_mask = (bt_normalized < threshold).astype(np.uint8)
    return cloud_mask, threshold


def process_single_file(nc_path, output_images_dir, output_masks_dir, tile_size=256, file_idx=0):
    """
    Process a single GOES-18 NetCDF file into image tiles and cloud masks.
    
    Returns number of tiles created.
    """
    import netCDF4 as nc
    from PIL import Image
    
    print(f"  Processing: {os.path.basename(nc_path)}")
    
    try:
        ds = nc.Dataset(nc_path, 'r')
    except Exception as e:
        print(f"  ERROR: Could not open {nc_path}: {e}")
        return 0
    
    try:
        # Read radiance data
        rad = ds.variables['Rad'][:]
        
        # Get Planck coefficients from the file
        planck_fk1 = ds.variables['planck_fk1'][:]
        planck_fk2 = ds.variables['planck_fk2'][:]
        planck_bc1 = ds.variables['planck_bc1'][:]
        planck_bc2 = ds.variables['planck_bc2'][:]
        
        # Convert to brightness temperature
        # Mask invalid values
        rad_masked = np.ma.masked_less_equal(rad, 0)
        bt = radiance_to_brightness_temp(rad_masked, planck_fk1, planck_fk2, planck_bc1, planck_bc2)
        
        # Fill masked values with NaN, then replace with median
        bt_filled = np.ma.filled(bt, fill_value=np.nan)
        median_val = np.nanmedian(bt_filled)
        bt_filled = np.nan_to_num(bt_filled, nan=median_val)
        
        # Normalize to [0, 1]
        bt_norm = normalize_brightness_temp(bt_filled)
        
        # Create cloud mask
        cloud_mask, otsu_thresh = create_cloud_mask(bt_norm)
        
        h, w = bt_norm.shape
        print(f"    Image size: {w}x{h}, Otsu threshold: {otsu_thresh:.3f}")
        
        # Create tiles
        tile_count = 0
        for row in range(0, h - tile_size + 1, tile_size):
            for col in range(0, w - tile_size + 1, tile_size):
                tile_bt = bt_norm[row:row+tile_size, col:col+tile_size]
                tile_mask = cloud_mask[row:row+tile_size, col:col+tile_size]
                
                # Skip tiles that are all NaN or all same value
                if np.std(tile_bt) < 0.01:
                    continue
                
                # Convert to 3-channel image (replicate IR across RGB for pre-trained encoder)
                tile_rgb = np.stack([tile_bt, tile_bt, tile_bt], axis=-1)
                tile_rgb = (tile_rgb * 255).astype(np.uint8)
                
                # Save image tile
                img_name = f"tile_{file_idx:04d}_{tile_count:04d}.png"
                img = Image.fromarray(tile_rgb)
                img.save(os.path.join(output_images_dir, img_name))
                
                # Save mask tile (0=clear, 255=cloud for visibility)
                mask_name = f"tile_{file_idx:04d}_{tile_count:04d}.png"
                mask_img = Image.fromarray(tile_mask * 255)
                mask_img.save(os.path.join(output_masks_dir, mask_name))
                
                tile_count += 1
        
        cloud_pct = np.mean(cloud_mask) * 100
        print(f"    Created {tile_count} tiles, cloud coverage: {cloud_pct:.1f}%")
        
        return tile_count
        
    finally:
        ds.close()


def create_train_val_split(images_dir, masks_dir, output_base, val_ratio=0.2):
    """
    Split processed data into train and validation sets.
    """
    import shutil
    import random
    
    image_files = sorted(os.listdir(images_dir))
    random.seed(42)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * (1 - val_ratio))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Create directories
    for split in ['train', 'val']:
        for subdir in ['images', 'masks']:
            os.makedirs(os.path.join(output_base, split, subdir), exist_ok=True)
    
    # Copy files
    for f in train_files:
        shutil.copy2(os.path.join(images_dir, f), os.path.join(output_base, 'train', 'images', f))
        shutil.copy2(os.path.join(masks_dir, f), os.path.join(output_base, 'train', 'masks', f))
    
    for f in val_files:
        shutil.copy2(os.path.join(images_dir, f), os.path.join(output_base, 'val', 'images', f))
        shutil.copy2(os.path.join(masks_dir, f), os.path.join(output_base, 'val', 'masks', f))
    
    print(f"\n  Train: {len(train_files)} tiles")
    print(f"  Val:   {len(val_files)} tiles")
    
    return len(train_files), len(val_files)


def main():
    print("=" * 60)
    print("AtmosGen — GOES-18 Data Processing Pipeline")
    print("=" * 60)
    
    # Paths
    raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    images_dir = os.path.join(processed_dir, 'all_images')
    masks_dir = os.path.join(processed_dir, 'all_masks')
    
    # Create output directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Find NetCDF files
    nc_files = sorted(glob.glob(os.path.join(raw_dir, '*.nc')))
    
    if not nc_files:
        print(f"\nERROR: No .nc files found in {raw_dir}")
        print("Please add GOES-18 Band 13 NetCDF files to data/raw/")
        sys.exit(1)
    
    print(f"\nFound {len(nc_files)} NetCDF files in {raw_dir}")
    print(f"Output: {processed_dir}\n")
    
    # Process each file
    total_tiles = 0
    for idx, nc_file in enumerate(nc_files):
        tiles = process_single_file(nc_file, images_dir, masks_dir, tile_size=256, file_idx=idx)
        total_tiles += tiles
    
    print(f"\n{'=' * 60}")
    print(f"Total tiles created: {total_tiles}")
    
    if total_tiles == 0:
        print("ERROR: No tiles were created. Check your NetCDF files.")
        sys.exit(1)
    
    # Create train/val split
    print(f"\nCreating train/val split (80/20)...")
    create_train_val_split(images_dir, masks_dir, processed_dir)
    
    print(f"\n{'=' * 60}")
    print("Data processing complete!")
    print(f"  Images: {processed_dir}/train/images/ and val/images/")
    print(f"  Masks:  {processed_dir}/train/masks/ and val/masks/")
    print(f"\nNext step: python backend/scripts/train_cloud_model.py")


if __name__ == '__main__':
    main()
