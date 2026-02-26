import torch
import numpy as np
import cv2
import base64
import io
import time
import logging
from PIL import Image
from typing import List, Dict, Any
from fastapi import UploadFile

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_model'))

from models.unet import UNet
from models.diffusion import Diffusion
from config import Config
from sampling import sample

class ModelService:
    """Service for loading and running AtmosGen model inference"""
    
    def __init__(self):
        self.model = None
        self.diffusion = None
        self.config = None
        self.device = None
        self.logger = logging.getLogger(__name__)
        self._loaded = False
        
    async def load_model(self):
        """Load the trained model and initialize components"""
        try:
            self.logger.info("Loading AtmosGen model...")
            
            # Initialize config
            self.config = Config()
            self.device = self.config.DEVICE
            
            self.logger.info(f"Using device: {self.device}")
            
            # Initialize model
            self.model = UNet().to(self.device)
            self.diffusion = Diffusion(self.config.TIMESTEPS)
            
            # Load checkpoint if available
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path:
                self.logger.info(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
            else:
                self.logger.warning("No checkpoint found, using randomly initialized model")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Disable gradients for inference
            for param in self.model.parameters():
                param.requires_grad = False
                
            self._loaded = True
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _find_latest_checkpoint(self) -> str:
        """Find the latest checkpoint file"""
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
        
        if not os.path.exists(checkpoint_dir):
            return None
            
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        
        if not checkpoints:
            return None
            
        # Sort by epoch number (assuming format: atmosgen_epoch_X.pth)
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest = checkpoints[-1]
        
        return os.path.join(checkpoint_dir, latest)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._loaded and self.model is not None
    
    async def predict(self, files: List[UploadFile]) -> Dict[str, Any]:
        """
        Generate weather forecast from uploaded images
        
        Args:
            files: List of uploaded image files
            
        Returns:
            Dictionary containing generated image and metadata
        """
        start_time = time.time()
        
        try:
            # Process uploaded images
            images = await self._process_uploaded_images(files)
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.stack(images).unsqueeze(0).to(self.device)
            
            self.logger.info(f"Input tensor shape: {input_tensor.shape}")
            
            # Generate prediction
            with torch.no_grad():
                generated_frame = sample(
                    self.model, 
                    self.diffusion, 
                    input_tensor, 
                    self.device
                )
            
            # Convert result to base64 image
            generated_image_b64 = self._tensor_to_base64(generated_frame[0])
            
            processing_time = time.time() - start_time
            
            return {
                "generated_image": generated_image_b64,
                "processing_time": processing_time,
                "input_sequence_length": len(files)
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    async def _process_uploaded_images(self, files: List[UploadFile]) -> List[torch.Tensor]:
        """Process uploaded images into model input format"""
        processed_images = []
        
        for file in files:
            # Read image data
            image_data = await file.read()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Resize to model input size
            image_resized = cv2.resize(image_np, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            
            # Normalize to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Convert to tensor (C, H, W)
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
            
            processed_images.append(image_tensor)
        
        return processed_images
    
    def _tensor_to_base64(self, tensor: torch.Tensor) -> str:
        """Convert tensor to base64 encoded image"""
        # Move to CPU and convert to numpy
        image_np = tensor.detach().cpu().numpy()
        
        # Transpose from (C, H, W) to (H, W, C)
        image_np = np.transpose(image_np, (1, 2, 0))
        
        # Clip values to [0, 1] and convert to [0, 255]
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)
        
        # Convert to PIL Image
        image_pil = Image.fromarray(image_np)
        
        # Convert to base64
        buffer = io.BytesIO()
        image_pil.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_b64