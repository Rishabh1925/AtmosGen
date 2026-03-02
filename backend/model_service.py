import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
import io
import os
import time
import logging
from PIL import Image
from typing import List, Dict, Any
from fastapi import UploadFile
import torchvision.transforms as transforms

class SimpleWeatherNet(nn.Module):
    """Simple CNN for weather forecasting - same architecture as trained model"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ModelService:
    """Service for loading and running AtmosGen simple weather model"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        self.logger = logging.getLogger(__name__)
        self._loaded = False
        
    async def load_model(self):
        """Load the trained simple weather model"""
        try:
            self.logger.info("Loading AtmosGen Simple Weather Model...")
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Using device: {self.device}")
            
            # Initialize model
            self.model = SimpleWeatherNet().to(self.device)
            
            # Load trained weights
            checkpoint_path = self._find_simple_checkpoint()
            if checkpoint_path:
                self.logger.info(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'model_config' in checkpoint:
                        config = checkpoint['model_config']
                        self.logger.info(f"Model: {config.get('model_name', 'Unknown')}")
                        self.logger.info(f"Training samples: {config.get('training_samples', 'Unknown')}")
                        self.logger.info(f"Final loss: {config.get('final_loss', 'Unknown')}")
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.logger.warning("No simple model checkpoint found")
                return False
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Setup image transforms (same as training)
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Disable gradients for inference
            for param in self.model.parameters():
                param.requires_grad = False
                
            self._loaded = True
            self.logger.info("Simple weather model loaded successfully")
            
            # Log model info
            param_count = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model parameters: {param_count:,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load simple model: {e}")
            raise
    
    def _find_simple_checkpoint(self) -> str:
        """Find the simple weather model checkpoint"""
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
        simple_checkpoint = os.path.join(checkpoint_dir, 'atmosgen_simple_v1.pth')
        
        if os.path.exists(simple_checkpoint):
            return simple_checkpoint
        
        # Fallback to any available checkpoint
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                return os.path.join(checkpoint_dir, checkpoints[0])
        
        return None
    
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
                "model_type": "Simple Weather CNN"
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