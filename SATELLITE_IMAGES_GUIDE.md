# 🛰️ Satellite Images Guide for AtmosGen

## 📍 **Current Sample Images**

Your project already includes 15 sample satellite images located at:
```
core_model/data/satellite_images/
├── img_0000.png
├── img_0001.png
├── img_0002.png
├── ...
└── img_0014.png
```

**To use these images:**
1. Navigate to `core_model/data/satellite_images/` in Finder
2. Select 3-8 images in sequence (e.g., img_0000.png to img_0007.png)
3. Upload them to the web interface at http://localhost:3000

---

## 🌍 **Getting More Satellite Images**

### **1. NOAA GOES Satellite Data (Recommended)**
- **Website:** https://www.goes.noaa.gov/
- **Data Portal:** https://www.avl.class.noaa.gov/saa/products/welcome
- **Image Type:** Weather satellite imagery (infrared, visible, water vapor)
- **Format:** Usually NETCDF, can be converted to PNG
- **Coverage:** Americas, Pacific

**How to access:**
1. Visit NOAA CLASS (Comprehensive Large Array-data Stewardship System)
2. Register for free account
3. Browse GOES-16/17/18 data
4. Download recent imagery
5. Convert to PNG using tools like `gdal` or Python

### **2. NASA Worldview**
- **Website:** https://worldview.earthdata.nasa.gov/
- **Features:** Interactive satellite imagery viewer
- **Data Sources:** MODIS, VIIRS, GOES, and more
- **Export:** Can download images directly as PNG/JPEG

**How to use:**
1. Go to NASA Worldview
2. Select layers (e.g., "Corrected Reflectance (True Color)")
3. Choose date/time
4. Use "Camera" icon to download images
5. Create time sequences by downloading multiple timestamps

### **3. European Space Agency (ESA) Copernicus**
- **Website:** https://scihub.copernicus.eu/
- **Satellites:** Sentinel-1, Sentinel-2, Sentinel-3
- **Coverage:** Global
- **Registration:** Free account required

### **4. USGS EarthExplorer**
- **Website:** https://earthexplorer.usgs.gov/
- **Satellites:** Landsat, MODIS, ASTER
- **Historical Data:** Extensive archive
- **Format:** Various formats, can export as images

### **5. Real-time Weather Satellite Loops**
- **NOAA:** https://www.star.nesdis.noaa.gov/GOES/
- **Weather Underground:** https://www.wunderground.com/maps/satellite
- **Windy.com:** https://www.windy.com/ (has satellite overlay)

---

## 🔧 **Converting Satellite Data to AtmosGen Format**

### **Requirements for AtmosGen:**
- **Format:** PNG or JPEG
- **Size:** Any size (automatically resized to 128x128)
- **Channels:** RGB (3 channels)
- **Sequence:** 3-10 images in temporal order

### **Python Script for Converting NETCDF to PNG:**
```python
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def convert_netcdf_to_png(netcdf_path, output_path):
    # Load satellite data
    ds = xr.open_dataset(netcdf_path)
    
    # Extract the main data variable (adjust based on your data)
    data = ds.CMI.values  # For GOES data, adjust variable name
    
    # Normalize to 0-255 range
    data_norm = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    
    # Convert to RGB (repeat grayscale across 3 channels)
    rgb_data = np.stack([data_norm, data_norm, data_norm], axis=-1)
    
    # Save as PNG
    Image.fromarray(rgb_data).save(output_path)

# Usage
convert_netcdf_to_png('goes_data.nc', 'satellite_image.png')
```

### **Batch Download Script for NASA Worldview:**
```python
import requests
from datetime import datetime, timedelta
import time

def download_worldview_sequence(start_date, num_images, output_dir):
    base_url = "https://worldview.earthdata.nasa.gov/api/v1/snapshot"
    
    for i in range(num_images):
        date = start_date + timedelta(hours=i)
        date_str = date.strftime("%Y-%m-%d")
        
        params = {
            'REQUEST': 'GetSnapshot',
            'TIME': date_str,
            'BBOX': '-140,20,-60,60',  # North America
            'CRS': 'EPSG:4326',
            'LAYERS': 'MODIS_Terra_CorrectedReflectance_TrueColor',
            'WIDTH': '512',
            'HEIGHT': '512',
            'FORMAT': 'image/png'
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            with open(f'{output_dir}/satellite_{i:04d}.png', 'wb') as f:
                f.write(response.content)
            print(f"Downloaded image {i+1}/{num_images}")
        
        time.sleep(1)  # Be respectful to the API

# Usage
start = datetime(2024, 1, 15, 12, 0)  # Start date/time
download_worldview_sequence(start, 8, 'satellite_sequence')
```

---

## 🎯 **Quick Start with Existing Images**

**For immediate testing:**

1. **Open Finder** and navigate to:
   ```
   /path/to/your/project/core_model/data/satellite_images/
   ```

2. **Select a sequence** (e.g., img_0000.png through img_0007.png)

3. **Open AtmosGen** in browser: http://localhost:3000

4. **Upload the images** using the file selector

5. **Generate forecast** and see the results!

---

## 📊 **Best Practices for Image Sequences**

### **Temporal Spacing:**
- **Ideal:** 15-30 minutes between images
- **Acceptable:** 1-6 hours between images
- **Avoid:** Random timestamps or gaps > 12 hours

### **Geographic Consistency:**
- Use images from the same geographic region
- Maintain consistent zoom level/resolution
- Avoid mixing different satellite sensors

### **Weather Events:**
- **Good for testing:** Storm systems, cloud formations
- **Interesting:** Hurricane development, frontal systems
- **Challenging:** Clear sky conditions (less to predict)

### **Image Quality:**
- Avoid heavily processed or filtered images
- Use raw or minimally processed satellite data
- Ensure good contrast and detail

---

## 🔍 **Example Data Sources by Region**

### **North America:**
- GOES-16 (GOES East)
- GOES-18 (GOES West)
- NOAA AVHRR

### **Europe:**
- Meteosat (MSG series)
- Sentinel-3 OLCI/SLSTR

### **Asia-Pacific:**
- Himawari-8/9 (Japan)
- FY-4A (China)

### **Global:**
- MODIS (Terra/Aqua)
- VIIRS (Suomi NPP)

---

## 🚀 **Ready to Test!**

You can start testing immediately with the provided sample images, then expand to real satellite data as needed. The system is designed to work with any RGB satellite imagery!

**Happy forecasting! 🌤️**