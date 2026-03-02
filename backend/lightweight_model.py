"""
Lightweight model service for deployment without large checkpoint files
"""
import torch
import torch.nn as nn
import numpy as np
import base64
import io
import time
import logging
from PIL import Image
from typing import List, Dict, Any
from fastapi import UploadFile
import torchvision.transforms as transforms

class LightweightWeatherNet(nn.Module):
    """Lightweight CNN for weather forecasting - works without checkpoints"""
    def __init__(self):
        super().__init__()
        # Simple encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Simple processing that creates weather-like patterns
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Add some weather-like effects
        weather_effect = torch.randn_like(decoded) * 0.1
        result = torch.clamp(decoded + weather_effect, 0, 1)
        
        return result

class LightweightModelService:
    """Lightweight model service that works without checkpoint files"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        self.logger = logging.getLogger(__name__)
        self._loaded = False
        
    async def load_model(self):
        """Load the lightweight model (no checkpoint needed)"""
        try:
            self.logger.info("Loading Lightweight Weather Model...")
            
            # Set device (CPU only for deployment)
            self.device = torch.device('cpu')
            self.logger.info(f"Using device: {self.device}")
            
            # Initialize model with random weights
            self.model = LightweightWeatherNet().to(self.device)
            self.model.eval()
            
            # Setup image transforms
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Smaller for faster processing
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Disable gradients for inference
            for param in self.model.parameters():
                param.requires_grad = False
                
            self._loaded = True
            self.logger.info("Lightweight weather model loaded successfully")
            
            # Log model info
            param_count = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model parameters: {param_count:,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load lightweight model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._loaded and self.model is not None
    
    async def predict(self, files: List[UploadFile]) -> Dict[str, Any]:
        """
        Generate weather forecast from uploaded image
        
        Args:
            files: List of uploaded image files (uses first image)
            
        Returns:
            Dictionary containing generated image and metadata
        """
        start_time = time.time()
        
        try:
            if not files:
                raise ValueError("No input files provided")
            
            # Process the first uploaded image
            input_image = await self._process_uploaded_image(files[0])
            
            self.logger.info(f"Input tensor shape: {input_image.shape}")
            
            # Generate prediction
            with torch.no_grad():
                prediction = self.model(input_image)
            
            # Convert result to base64 image
            generated_image_b64 = self._tensor_to_base64(prediction[0])
            
            processing_time = time.time() - start_time
            
            return {
                "generated_image": generated_image_b64,
                "processing_time": processing_time,
                "input_sequence_length": 1,
                "model_type": "Lightweight Weather CNN",
                "note": "Demo model - for production use, train with real weather data"
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    async def _process_uploaded_image(self, file: UploadFile) -> torch.Tensor:
        """Process uploaded image into model input format"""
        # Read image data
        image_data = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms and add batch dimension
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return image_tensor
    
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