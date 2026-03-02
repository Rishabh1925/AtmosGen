import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from PIL import Image
import io
import base64
import os

logger = logging.getLogger(__name__)

class SatelliteService:
    """Service for fetching real satellite data from NASA and NOAA APIs"""
    
    def __init__(self):
        self.nasa_worldview_base = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best"
        self.noaa_goes_base = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD"
        
        # Common satellite layers
        self.layers = {
            "visible": "MODIS_Aqua_CorrectedReflectance_TrueColor",
            "infrared": "MODIS_Aqua_CorrectedReflectance_Bands721", 
            "water_vapor": "AIRS_L2_Water_Vapor_Ocean_Day",
            "cloud_top": "MODIS_Aqua_Cloud_Top_Temp_Day"
        }
        
        # Predefined regions
        self.regions = {
            "north_america": {"bbox": [-140, 20, -50, 60], "name": "North America"},
            "europe": {"bbox": [-20, 35, 40, 70], "name": "Europe"},
            "asia_pacific": {"bbox": [100, 10, 160, 50], "name": "Asia Pacific"},
            "global": {"bbox": [-180, -90, 180, 90], "name": "Global"}
        }
    
    async def get_available_dates(self, days_back: int = 7) -> List[str]:
        """Get list of available dates for satellite data"""
        dates = []
        today = datetime.now()
        
        for i in range(days_back):
            date = today - timedelta(days=i)
            dates.append(date.strftime("%Y-%m-%d"))
        
        return dates
    
    async def fetch_nasa_worldview_image(
        self, 
        layer: str, 
        date: str, 
        bbox: List[float], 
        width: int = 512, 
        height: int = 512
    ) -> Optional[bytes]:
        """
        Fetch satellite image from NASA Worldview API
        
        Args:
            layer: Satellite layer (visible, infrared, etc.)
            date: Date in YYYY-MM-DD format
            bbox: Bounding box [west, south, east, north]
            width: Image width in pixels
            height: Image height in pixels
        """
        try:
            layer_name = self.layers.get(layer, self.layers["visible"])
            
            # NASA Worldview WMTS URL
            url = f"{self.nasa_worldview_base}/{layer_name}/default/{date}/EPSG4326_250m/{{}}/{{}}/{{}}.jpg"
            
            # Calculate tile coordinates for the bounding box
            # For simplicity, we'll use a single tile approach
            # In production, you'd implement proper tile calculation
            
            # Use GetMap request instead for easier bbox handling
            wms_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
            
            params = {
                "SERVICE": "WMS",
                "VERSION": "1.3.0",
                "REQUEST": "GetMap",
                "LAYERS": layer_name,
                "CRS": "EPSG:4326",
                "BBOX": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",  # south,west,north,east for WMS 1.3.0
                "WIDTH": width,
                "HEIGHT": height,
                "FORMAT": "image/jpeg",
                "TIME": date
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(wms_url, params=params, timeout=30) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        logger.info(f"Successfully fetched NASA image for {date}, layer {layer}")
                        return image_data
                    else:
                        logger.error(f"NASA API error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Failed to fetch NASA image: {e}")
            return None
    
    async def fetch_noaa_goes_image(
        self, 
        date: str, 
        channel: str = "GEOCOLOR"
    ) -> Optional[bytes]:
        """
        Fetch GOES satellite image from NOAA
        
        Args:
            date: Date in YYYY-MM-DD format
            channel: GOES channel (GEOCOLOR, Band02, etc.)
        """
        try:
            # NOAA GOES image URL pattern
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            year = date_obj.year
            day_of_year = date_obj.timetuple().tm_yday
            
            # Try to get the latest image for that date
            # NOAA updates images every 10-15 minutes
            for hour in [12, 18, 6, 0]:  # Try different times
                url = f"{self.noaa_goes_base}/{channel}/{year}/{day_of_year:03d}/{hour:02d}/"
                
                async with aiohttp.ClientSession() as session:
                    # First get the directory listing to find available files
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            # Parse HTML to find image files (simplified)
                            content = await response.text()
                            # Look for .jpg files in the response
                            import re
                            jpg_files = re.findall(r'href="([^"]*\.jpg)"', content)
                            
                            if jpg_files:
                                # Get the latest file
                                image_url = url + jpg_files[-1]
                                
                                async with session.get(image_url, timeout=30) as img_response:
                                    if img_response.status == 200:
                                        image_data = await img_response.read()
                                        logger.info(f"Successfully fetched NOAA GOES image for {date}")
                                        return image_data
            
            logger.warning(f"No NOAA GOES image found for {date}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch NOAA image: {e}")
            return None
    
    async def get_satellite_sequence(
        self, 
        region: str, 
        layer: str = "visible", 
        sequence_length: int = 4,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get a sequence of satellite images for ML prediction
        
        Args:
            region: Region name (north_america, europe, etc.)
            layer: Satellite layer type
            sequence_length: Number of images in sequence
            end_date: End date for sequence (defaults to today)
        """
        try:
            if region not in self.regions:
                raise ValueError(f"Unknown region: {region}")
            
            bbox = self.regions[region]["bbox"]
            
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            
            images = []
            
            # Fetch sequence of images (daily intervals)
            for i in range(sequence_length):
                date_obj = end_date_obj - timedelta(days=i)
                date_str = date_obj.strftime("%Y-%m-%d")
                
                # Try NASA first, fallback to NOAA if needed
                image_data = await self.fetch_nasa_worldview_image(
                    layer=layer,
                    date=date_str,
                    bbox=bbox
                )
                
                if image_data:
                    # Convert to base64 for frontend
                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                    
                    images.append({
                        "date": date_str,
                        "layer": layer,
                        "region": region,
                        "image_data": f"data:image/jpeg;base64,{image_b64}",
                        "source": "NASA Worldview"
                    })
                else:
                    logger.warning(f"Could not fetch image for {date_str}")
            
            # Reverse to get chronological order (oldest first)
            images.reverse()
            
            logger.info(f"Retrieved {len(images)} satellite images for {region}")
            return images
            
        except Exception as e:
            logger.error(f"Failed to get satellite sequence: {e}")
            return []
    
    async def get_regions_list(self) -> Dict[str, Dict]:
        """Get list of available regions"""
        return self.regions
    
    async def get_layers_list(self) -> Dict[str, str]:
        """Get list of available satellite layers"""
        return {
            "visible": "True Color (Visible Light)",
            "infrared": "False Color Infrared", 
            "water_vapor": "Water Vapor",
            "cloud_top": "Cloud Top Temperature"
        }
    
    def preprocess_for_ml(self, image_data: bytes) -> Image.Image:
        """
        Preprocess satellite image for ML model
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            PIL Image ready for ML processing
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Resize to model input size (assuming 256x256)
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, bool]:
        """Check if satellite data services are available"""
        health = {
            "nasa_worldview": False,
            "noaa_goes": False
        }
        
        try:
            # Test NASA Worldview
            async with aiohttp.ClientSession() as session:
                nasa_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities"
                async with session.get(nasa_url, timeout=10) as response:
                    health["nasa_worldview"] = response.status == 200
        except:
            pass
        
        try:
            # Test NOAA GOES
            async with aiohttp.ClientSession() as session:
                noaa_url = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/"
                async with session.get(noaa_url, timeout=10) as response:
                    health["noaa_goes"] = response.status == 200
        except:
            pass
        
        return health