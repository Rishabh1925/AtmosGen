#!/usr/bin/env python3
"""
MOSDAC Data Pipeline Setup
Download and process real satellite data for training production weather model
"""

import os
import requests
import numpy as np
from datetime import datetime, timedelta
import h5py
import netCDF4 as nc
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import cv2
from PIL import Image

class MOSDACDataPipeline:
    """Pipeline for downloading and processing MOSDAC satellite data"""
    
    def __init__(self, data_dir: str = "data/mosdac"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # MOSDAC data configuration
        self.base_url = "https://mosdac.gov.in/data"  # Update with actual MOSDAC API
        self.channels = {
            'VIS': 'visible',
            'IR1': 'infrared_3.9',
            'IR2': 'infrared_10.8', 
            'IR3': 'infrared_12.0',
            'WV': 'water_vapor_6.7'
        }
        
        self.logger = logging.getLogger(__name__)
        
    def download_satellite_data(self, 
                               start_date: datetime, 
                               end_date: datetime,
                               region: str = "india") -> List[str]:
        """
        Download INSAT-3D/3DR satellite imagery from MOSDAC
        
        Args:
            start_date: Start date for data download
            end_date: End date for data download  
            region: Geographic region (default: india)
            
        Returns:
            List of downloaded file paths
        """
        downloaded_files = []
        current_date = start_date
        
        while current_date <= end_date:
            for hour in range(0, 24, 3):  # Every 3 hours
                timestamp = current_date.replace(hour=hour, minute=0, second=0)
                
                for channel, channel_name in self.channels.items():
                    try:
                        file_path = self._download_single_file(timestamp, channel, region)
                        if file_path:
                            downloaded_files.append(file_path)
                            self.logger.info(f"Downloaded: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to download {channel} for {timestamp}: {e}")
                        
            current_date += timedelta(days=1)
            
        return downloaded_files
    
    def _download_single_file(self, 
                             timestamp: datetime, 
                             channel: str, 
                             region: str) -> str:
        """Download a single satellite file"""
        
        # Construct filename based on MOSDAC naming convention
        filename = f"3DIMG_{timestamp.strftime('%d%b%Y_%H%M')}_{channel}_{region.upper()}.h5"
        file_path = self.data_dir / filename
        
        # Skip if already exists
        if file_path.exists():
            return str(file_path)
            
        # Construct download URL (update with actual MOSDAC API)
        url = f"{self.base_url}/{timestamp.year}/{timestamp.month:02d}/{filename}"
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return str(file_path)
    
    def process_satellite_imagery(self, file_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Process downloaded satellite files into training-ready format
        
        Args:
            file_paths: List of downloaded HDF5/NetCDF files
            
        Returns:
            Dictionary with processed imagery arrays
        """
        processed_data = {
            'images': [],
            'timestamps': [],
            'channels': [],
            'metadata': []
        }
        
        for file_path in file_paths:
            try:
                data = self._process_single_file(file_path)
                if data:
                    processed_data['images'].append(data['image'])
                    processed_data['timestamps'].append(data['timestamp'])
                    processed_data['channels'].append(data['channel'])
                    processed_data['metadata'].append(data['metadata'])
                    
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                
        return processed_data
    
    def _process_single_file(self, file_path: str) -> Dict:
        """Process a single satellite file"""
        
        file_path = Path(file_path)
        
        # Determine file type and process accordingly
        if file_path.suffix == '.h5':
            return self._process_hdf5_file(file_path)
        elif file_path.suffix == '.nc':
            return self._process_netcdf_file(file_path)
        else:
            self.logger.warning(f"Unsupported file format: {file_path}")
            return None
    
    def _process_hdf5_file(self, file_path: Path) -> Dict:
        """Process HDF5 satellite file"""
        
        with h5py.File(file_path, 'r') as f:
            # Extract image data (update keys based on actual MOSDAC structure)
            if 'IMG' in f:
                image_data = f['IMG'][:]
            elif 'image' in f:
                image_data = f['image'][:]
            else:
                # Find the main data array
                keys = list(f.keys())
                image_data = f[keys[0]][:]
            
            # Extract metadata
            metadata = {}
            if 'metadata' in f:
                for key in f['metadata'].keys():
                    metadata[key] = f['metadata'][key][()]
                    
            # Normalize image data
            image_normalized = self._normalize_satellite_image(image_data)
            
            # Extract timestamp from filename
            timestamp = self._extract_timestamp_from_filename(file_path.name)
            
            # Extract channel from filename
            channel = self._extract_channel_from_filename(file_path.name)
            
            return {
                'image': image_normalized,
                'timestamp': timestamp,
                'channel': channel,
                'metadata': metadata
            }
    
    def _process_netcdf_file(self, file_path: Path) -> Dict:
        """Process NetCDF satellite file"""
        
        with nc.Dataset(file_path, 'r') as f:
            # Extract main variable (update based on actual structure)
            var_names = list(f.variables.keys())
            main_var = None
            
            for var_name in ['radiance', 'brightness_temperature', 'reflectance']:
                if var_name in var_names:
                    main_var = var_name
                    break
                    
            if not main_var:
                main_var = var_names[0]  # Use first variable
                
            image_data = f.variables[main_var][:]
            
            # Extract metadata
            metadata = {}
            for attr in f.ncattrs():
                metadata[attr] = getattr(f, attr)
                
            # Normalize image data
            image_normalized = self._normalize_satellite_image(image_data)
            
            # Extract timestamp and channel
            timestamp = self._extract_timestamp_from_filename(file_path.name)
            channel = self._extract_channel_from_filename(file_path.name)
            
            return {
                'image': image_normalized,
                'timestamp': timestamp,
                'channel': channel,
                'metadata': metadata
            }
    
    def _normalize_satellite_image(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize satellite imagery for training"""
        
        # Handle different data types
        if image_data.dtype == np.uint16:
            image_data = image_data.astype(np.float32) / 65535.0
        elif image_data.dtype == np.uint8:
            image_data = image_data.astype(np.float32) / 255.0
        else:
            # Normalize to [0, 1] range
            image_data = image_data.astype(np.float32)
            image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
        
        # Resize to standard size (512x512)
        if len(image_data.shape) == 2:
            image_resized = cv2.resize(image_data, (512, 512))
        else:
            # Handle multi-channel data
            image_resized = np.zeros((512, 512, image_data.shape[-1]))
            for i in range(image_data.shape[-1]):
                image_resized[:, :, i] = cv2.resize(image_data[:, :, i], (512, 512))
        
        return image_resized
    
    def _extract_timestamp_from_filename(self, filename: str) -> datetime:
        """Extract timestamp from MOSDAC filename"""
        # Example: 3DIMG_01JAN2024_1200_VIS_INDIA.h5
        try:
            parts = filename.split('_')
            date_str = parts[1]  # 01JAN2024
            time_str = parts[2]  # 1200
            
            # Parse date
            day = int(date_str[:2])
            month_str = date_str[2:5]
            year = int(date_str[5:])
            
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            month = month_map[month_str]
            
            # Parse time
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            
            return datetime(year, month, day, hour, minute)
            
        except Exception as e:
            self.logger.error(f"Failed to parse timestamp from {filename}: {e}")
            return datetime.now()
    
    def _extract_channel_from_filename(self, filename: str) -> str:
        """Extract channel information from filename"""
        for channel in self.channels.keys():
            if channel in filename:
                return channel
        return 'UNKNOWN'
    
    def create_training_sequences(self, 
                                processed_data: Dict, 
                                sequence_length: int = 4,
                                forecast_steps: int = 3) -> Dict:
        """
        Create training sequences for temporal forecasting
        
        Args:
            processed_data: Processed satellite imagery data
            sequence_length: Number of input frames
            forecast_steps: Number of future frames to predict
            
        Returns:
            Training sequences with inputs and targets
        """
        
        # Sort data by timestamp
        sorted_indices = np.argsort(processed_data['timestamps'])
        
        sequences = {
            'inputs': [],
            'targets': [],
            'timestamps': []
        }
        
        # Create sequences
        for i in range(len(sorted_indices) - sequence_length - forecast_steps + 1):
            # Input sequence
            input_frames = []
            for j in range(sequence_length):
                idx = sorted_indices[i + j]
                input_frames.append(processed_data['images'][idx])
            
            # Target sequence
            target_frames = []
            for j in range(forecast_steps):
                idx = sorted_indices[i + sequence_length + j]
                target_frames.append(processed_data['images'][idx])
            
            sequences['inputs'].append(np.stack(input_frames))
            sequences['targets'].append(np.stack(target_frames))
            sequences['timestamps'].append(processed_data['timestamps'][sorted_indices[i]])
        
        return sequences
    
    def save_training_data(self, sequences: Dict, output_path: str):
        """Save processed training data"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_path,
            inputs=np.array(sequences['inputs']),
            targets=np.array(sequences['targets']),
            timestamps=sequences['timestamps']
        )
        
        self.logger.info(f"Training data saved to {output_path}")

def main():
    """Main function to run MOSDAC data pipeline"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline
    pipeline = MOSDACDataPipeline()
    
    # Download data for last 3 months (adjust as needed)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    print(f"🛰️ Starting MOSDAC data download from {start_date} to {end_date}")
    
    # Download satellite data
    file_paths = pipeline.download_satellite_data(start_date, end_date)
    print(f"📥 Downloaded {len(file_paths)} files")
    
    # Process imagery
    processed_data = pipeline.process_satellite_imagery(file_paths)
    print(f"🔄 Processed {len(processed_data['images'])} images")
    
    # Create training sequences
    sequences = pipeline.create_training_sequences(processed_data)
    print(f"📊 Created {len(sequences['inputs'])} training sequences")
    
    # Save training data
    pipeline.save_training_data(sequences, "data/mosdac_training_data.npz")
    print("✅ Training data ready!")
    
    print("\n🚀 Ready to train production weather model with real satellite data!")

if __name__ == "__main__":
    main()