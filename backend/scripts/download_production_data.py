#!/usr/bin/env python3
"""
Download real NOAA GOES satellite data for production training
This creates a legitimate, high-quality weather forecasting dataset
"""

import os
import requests
import boto3
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

class NOAADataDownloader:
    """Download and process real NOAA GOES satellite data"""
    
    def __init__(self, output_dir="production_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # NOAA GOES AWS buckets (public, no auth needed)
        self.buckets = {
            'GOES-16': 'noaa-goes16',
            'GOES-17': 'noaa-goes17', 
            'GOES-18': 'noaa-goes18'
        }
        
        # Weather-relevant channels
        self.channels = {
            'C02': 'Red (0.64 μm) - Clouds, vegetation',
            'C07': 'Shortwave IR (3.9 μm) - Low clouds, fog',
            'C08': 'Water vapor (6.2 μm) - Upper moisture',
            'C09': 'Water vapor (6.9 μm) - Mid moisture', 
            'C10': 'Water vapor (7.3 μm) - Lower moisture',
            'C13': 'Clean IR (10.3 μm) - Surface/cloud temp',
            'C14': 'IR (11.2 μm) - Cloud imagery',
            'C15': 'Dirty IR (12.3 μm) - Cloud particles'
        }
        
        # Initialize S3 client (no credentials needed for public buckets)
        self.s3_client = boto3.client('s3', 
                                     region_name='us-east-1',
                                     aws_access_key_id='',
                                     aws_secret_access_key='')
    
    def get_available_dates(self, satellite='GOES-16', year=2024):
        """Get available dates for a satellite"""
        
        bucket = self.buckets[satellite]
        prefix = f"ABI-L1b-RadC/{year}/"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                Delimiter='/'
            )
            
            dates = []
            if 'CommonPrefixes' in response:
                for obj in response['CommonPrefixes']:
                    date_str = obj['Prefix'].split('/')[-2]
                    if len(date_str) == 3:  # Day of year format
                        dates.append(int(date_str))
            
            return sorted(dates)
            
        except Exception as e:
            print(f"Error getting dates for {satellite}: {e}")
            return []
    
    def download_satellite_sequence(self, satellite, date, hour, channels=['C02', 'C13']):
        """Download a temporal sequence of satellite images"""
        
        bucket = self.buckets[satellite]
        year = 2024  # Focus on recent data
        
        # Create sequence directory
        seq_dir = self.output_dir / f"{satellite}_{year}_{date:03d}_{hour:02d}"
        seq_dir.mkdir(exist_ok=True)
        
        downloaded_files = []
        
        for channel in channels:
            channel_files = []
            
            # Download 4 consecutive images (20-minute sequence)
            for minute_offset in [0, 10, 20, 30]:
                minute = minute_offset
                
                # GOES file naming convention
                filename_pattern = f"OR_ABI-L1b-RadC-M6{channel}_{satellite[5:]}_s{year}{date:03d}{hour:02d}{minute:02d}"
                
                try:
                    # List files matching pattern
                    prefix = f"ABI-L1b-RadC/{year}/{date:03d}/{hour:02d}/"
                    
                    response = self.s3_client.list_objects_v2(
                        Bucket=bucket,
                        Prefix=prefix
                    )
                    
                    if 'Contents' in response:
                        # Find matching file
                        matching_files = [
                            obj['Key'] for obj in response['Contents']
                            if channel in obj['Key'] and f"{hour:02d}{minute:02d}" in obj['Key']
                        ]
                        
                        if matching_files:
                            # Download first matching file
                            s3_key = matching_files[0]
                            local_filename = os.path.basename(s3_key)
                            local_path = seq_dir / local_filename
                            
                            if not local_path.exists():
                                self.s3_client.download_file(bucket, s3_key, str(local_path))
                            
                            channel_files.append(str(local_path))
                
                except Exception as e:
                    print(f"Error downloading {channel} at {hour:02d}:{minute:02d}: {e}")
                    continue
            
            if len(channel_files) >= 3:  # Need at least 3 images for sequence
                downloaded_files.extend(channel_files)
        
        return downloaded_files
    
    def process_netcdf_to_image(self, netcdf_path, output_path):
        """Convert NetCDF satellite data to processed image"""
        
        try:
            # Open NetCDF file
            ds = xr.open_dataset(netcdf_path)
            
            # Get radiance data
            if 'Rad' in ds.variables:
                radiance = ds['Rad'].values
            else:
                print(f"No radiance data in {netcdf_path}")
                return None
            
            # Handle different shapes
            if len(radiance.shape) == 3:
                radiance = radiance[0]
            
            # Remove invalid values
            radiance = np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize to 0-255 range
            if np.max(radiance) > np.min(radiance):
                radiance_norm = ((radiance - np.min(radiance)) / 
                               (np.max(radiance) - np.min(radiance)) * 255).astype(np.uint8)
            else:
                radiance_norm = np.zeros_like(radiance, dtype=np.uint8)
            
            # Convert to PIL and save
            from PIL import Image
            
            # Resize to standard size for training
            image = Image.fromarray(radiance_norm, mode='L')
            image = image.convert('RGB')  # Convert to RGB for model compatibility
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            image.save(output_path, quality=95)
            
            # Clean up
            ds.close()
            
            return output_path
            
        except Exception as e:
            print(f"Error processing {netcdf_path}: {e}")
            return None
    
    def create_training_sequences(self, num_sequences=1000):
        """Create training sequences from downloaded data"""
        
        print(f"Creating {num_sequences} training sequences from real NOAA data...")
        
        # Get available dates for GOES-16 (most reliable)
        satellite = 'GOES-16'
        available_dates = self.get_available_dates(satellite, 2024)
        
        if not available_dates:
            print("No available dates found. Using alternative approach...")
            return self.create_sample_sequences()
        
        print(f"Found {len(available_dates)} available dates for {satellite}")
        
        sequences_created = 0
        training_data = []
        
        # Sample dates and hours for diversity
        selected_dates = np.random.choice(available_dates, 
                                        min(50, len(available_dates)), 
                                        replace=False)
        
        for date in tqdm(selected_dates, desc="Processing dates"):
            # Try different hours of the day
            for hour in [6, 12, 18]:  # Morning, noon, evening
                if sequences_created >= num_sequences:
                    break
                
                try:
                    # Download sequence
                    files = self.download_satellite_sequence(
                        satellite, date, hour, 
                        channels=['C02', 'C13']  # Visible and IR
                    )
                    
                    if len(files) >= 6:  # Need at least 6 files (2 channels × 3 times)
                        # Process NetCDF files to images
                        processed_files = []
                        
                        for nc_file in files:
                            img_path = str(nc_file).replace('.nc', '.jpg')
                            
                            if self.process_netcdf_to_image(nc_file, img_path):
                                processed_files.append(img_path)
                        
                        if len(processed_files) >= 4:  # Need 4 for sequence (3 input + 1 target)
                            # Create training sample
                            sample_data = {
                                'satellite': satellite,
                                'date': date,
                                'hour': hour,
                                'input_files': processed_files[:3],
                                'target_file': processed_files[3],
                                'description': f"Real NOAA {satellite} sequence from day {date}, hour {hour}"
                            }
                            
                            training_data.append(sample_data)
                            sequences_created += 1
                            
                            if sequences_created % 10 == 0:
                                print(f"Created {sequences_created}/{num_sequences} sequences")
                
                except Exception as e:
                    print(f"Error processing date {date}, hour {hour}: {e}")
                    continue
        
        # Save training metadata
        dataset_info = {
            'total_samples': len(training_data),
            'created_at': datetime.now().isoformat(),
            'description': 'Real NOAA GOES satellite data for weather forecasting',
            'satellite': satellite,
            'data_source': 'NOAA GOES AWS Public Dataset',
            'channels_used': list(self.channels.keys()),
            'samples': training_data
        }
        
        with open(self.output_dir / 'production_dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n Created {len(training_data)} real satellite training sequences")
        print(f" Data saved in: {self.output_dir}")
        
        return training_data
    
    def create_sample_sequences(self):
        """Create sample sequences using your existing NOAA data"""
        
        print("Creating training sequences from existing NOAA data...")
        
        # Use the NOAA files you already have
        noaa_files = list(Path("../data/raw").glob("*.nc"))
        
        if not noaa_files:
            print(" No NOAA .nc files found in ../data/raw/")
            return []
        
        print(f"Found {len(noaa_files)} existing NOAA files")
        
        training_data = []
        processed_dir = self.output_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # Process existing files into training sequences
        for i in range(0, len(noaa_files) - 3, 1):  # Sliding window
            sequence_files = noaa_files[i:i+4]
            
            processed_sequence = []
            
            for j, nc_file in enumerate(sequence_files):
                # Process to image
                img_filename = f"real_noaa_seq_{i:03d}_frame_{j}.jpg"
                img_path = processed_dir / img_filename
                
                if self.process_netcdf_to_image(nc_file, img_path):
                    processed_sequence.append(str(img_path))
            
            if len(processed_sequence) == 4:
                sample_data = {
                    'sequence_id': i,
                    'input_files': processed_sequence[:3],
                    'target_file': processed_sequence[3],
                    'description': f"Real NOAA GOES-18 sequence {i}",
                    'source_files': [str(f) for f in sequence_files]
                }
                
                training_data.append(sample_data)
        
        # Save metadata
        dataset_info = {
            'total_samples': len(training_data),
            'created_at': datetime.now().isoformat(),
            'description': 'Real NOAA GOES-18 satellite data sequences',
            'data_source': 'User-provided NOAA .nc files',
            'samples': training_data
        }
        
        with open(self.output_dir / 'production_dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f" Created {len(training_data)} real NOAA training sequences")
        return training_data

def download_era5_data():
    """Download ERA5 reanalysis data for additional training features"""
    
    print("ERA5 data download requires CDS API setup...")
    print("Visit: https://cds.climate.copernicus.eu/api-how-to")
    print("This provides additional atmospheric variables for training")
    
    # ERA5 download would require CDS API setup
    # For now, focus on GOES satellite data
    
    return None

def main():
    """Main function for production data download"""
    
    parser = argparse.ArgumentParser(description='Download real NOAA satellite data')
    parser.add_argument('--sequences', type=int, default=100, 
                       help='Number of training sequences to create')
    parser.add_argument('--output-dir', type=str, default='../data/production',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DOWNLOADING REAL NOAA SATELLITE DATA")
    print("=" * 60)
    
    # Initialize downloader
    downloader = NOAADataDownloader(args.output_dir)
    
    # Create training sequences
    training_data = downloader.create_training_sequences(args.sequences)
    
    if training_data:
        print("\n" + "=" * 60)
        print("REAL DATA DOWNLOAD COMPLETE!")
        print("=" * 60)
        print(f" Created {len(training_data)} real satellite sequences")
        print(f" Data location: {args.output_dir}")
        print(f"  Source: NOAA GOES satellite imagery")
        print(f" Ready for production training!")
        
        print("\nNext steps:")
        print("1. Package for Kaggle: python package_production_data.py")
        print("2. Train on Kaggle with real data")
        print("3. Achieve production-grade accuracy!")
    else:
        print("\n Failed to create training data")
        print("Check internet connection and try again")

if __name__ == "__main__":
    main()