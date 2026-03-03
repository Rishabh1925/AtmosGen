"""
Cloud Segmentation Model — Pre-trained U-Net with EfficientNet encoder.

Fine-tuned on GOES-18 Band 13 IR satellite data for cloud detection.
Uses segmentation-models-pytorch with ImageNet pre-trained encoder.
"""

import torch
import torch.nn as nn
import numpy as np
import base64
import io
import time
import logging
import os
from PIL import Image
from typing import Dict, Any, Optional
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class CloudModelService:
    """
    Service for loading and running the fine-tuned cloud segmentation model.
    
    Model: U-Net with EfficientNet-B0 encoder (pre-trained on ImageNet).
    Task: Binary cloud segmentation from IR satellite imagery.
    Output: Cloud mask + cloud coverage percentage.
    """
    
    def __init__(self):
        self.model = None
        self.device = None
        self._loaded = False
        self.model_info = {}
    
    async def load_model(self):
        """Load the trained cloud segmentation model."""
        try:
            import segmentation_models_pytorch as smp
            import torchvision.transforms as transforms
            
            logger.info("Loading Cloud Segmentation Model...")
            
            # Device selection: CUDA > MPS (Apple Silicon) > CPU
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
            
            logger.info(f"Using device: {self.device}")
            
            # Initialize model architecture (must match training)
            self.model = smp.Unet(
                encoder_name='efficientnet-b0',
                encoder_weights=None,  # We load our own weights
                in_channels=3,
                classes=1,
                activation=None  # We apply sigmoid manually
            ).to(self.device)
            
            # Find checkpoint
            checkpoint_path = self._find_checkpoint()
            
            if checkpoint_path:
                logger.info(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model_info = {
                        'best_iou': checkpoint.get('best_iou', 'N/A'),
                        'epoch': checkpoint.get('epoch', 'N/A'),
                        'encoder': 'EfficientNet-B0 (ImageNet)',
                        'architecture': 'U-Net',
                    }
                    logger.info(f"Checkpoint loaded — Best IoU: {self.model_info['best_iou']}")
                else:
                    self.model.load_state_dict(checkpoint)
                    self.model_info = {
                        'encoder': 'EfficientNet-B0 (ImageNet)',
                        'architecture': 'U-Net',
                    }
            else:
                logger.warning("No checkpoint found — using ImageNet pre-trained encoder only")
                # Re-init with ImageNet weights as fallback
                self.model = smp.Unet(
                    encoder_name='efficientnet-b0',
                    encoder_weights='imagenet',
                    in_channels=3,
                    classes=1,
                    activation=None
                ).to(self.device)
                self.model_info = {
                    'encoder': 'EfficientNet-B0 (ImageNet)',
                    'architecture': 'U-Net',
                    'note': 'Using pre-trained weights only (not fine-tuned)'
                }
            
            self.model.eval()
            
            # Disable gradients
            for param in self.model.parameters():
                param.requires_grad = False
            
            self._loaded = True
            
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded — {param_count:,} parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cloud model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _find_checkpoint(self) -> Optional[str]:
        """Find the best cloud segmentation checkpoint."""
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
        
        # Priority order
        candidates = [
            'cloud_seg_best.pth',
            'cloud_seg_v1.pth',
        ]
        
        for name in candidates:
            path = os.path.join(checkpoint_dir, name)
            if os.path.exists(path):
                return path
        
        # Fallback: any cloud_seg checkpoint
        if os.path.exists(checkpoint_dir):
            for f in sorted(os.listdir(checkpoint_dir)):
                if f.startswith('cloud_seg') and f.endswith('.pth'):
                    return os.path.join(checkpoint_dir, f)
        
        return None
    
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None
    
    async def predict(self, files: list) -> Dict[str, Any]:
        """
        Run cloud segmentation on uploaded satellite image.
        
        Returns:
            - cloud_mask: base64 encoded cloud mask image
            - cloud_coverage_pct: percentage of image covered by clouds
            - generated_image: base64 encoded overlay visualization
            - processing_time: inference time in seconds
            - model_info: model metadata
        """
        import torchvision.transforms as transforms
        
        start_time = time.time()
        
        if not files:
            raise ValueError("No input files provided")
        
        # Read and preprocess image
        file = files[0]
        if hasattr(file, 'read'):
            image_data = await file.read()
        else:
            image_data = file
        
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        original_size = image.size
        
        # Resize for model input
        input_size = (256, 256)
        image_resized = image.resize(input_size, Image.Resampling.LANCZOS)
        
        # Transform to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image_resized).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            prob_mask = torch.sigmoid(logits)
            binary_mask = (prob_mask > 0.5).float()
        
        # Calculate cloud coverage
        cloud_coverage_pct = float(binary_mask.mean().item() * 100)
        
        # Convert mask to numpy
        mask_np = binary_mask[0, 0].cpu().numpy()
        prob_np = prob_mask[0, 0].cpu().numpy()
        
        # Create visualization: overlay cloud mask on input image
        overlay_image = self._create_overlay(image_resized, mask_np, prob_np)
        
        # Encode outputs as base64
        cloud_mask_b64 = self._array_to_base64(mask_np)
        overlay_b64 = self._pil_to_base64(overlay_image)
        
        # Also encode the input for side-by-side display
        input_b64 = self._pil_to_base64(image_resized)
        
        processing_time = time.time() - start_time
        
        return {
            "generated_image": overlay_b64,  # Main visualization
            "cloud_mask": cloud_mask_b64,     # Raw binary mask
            "input_image": input_b64,         # Input for comparison
            "cloud_coverage_pct": round(cloud_coverage_pct, 1),
            "processing_time": round(processing_time, 3),
            "model_type": "Cloud Segmentation (U-Net + EfficientNet-B0)",
            "model_info": self.model_info,
        }
    
    def _create_overlay(self, image: Image.Image, mask: np.ndarray, prob: np.ndarray) -> Image.Image:
        """Create a visualization overlaying cloud mask on the satellite image."""
        img_np = np.array(image).astype(np.float32)
        
        # Create colored overlay: clouds in blue-white, clear in transparent
        overlay = img_np.copy()
        
        # Cloud regions: tint blue-white
        cloud_pixels = mask > 0.5
        overlay[cloud_pixels, 0] = overlay[cloud_pixels, 0] * 0.4 + 180 * 0.6  # R
        overlay[cloud_pixels, 1] = overlay[cloud_pixels, 1] * 0.4 + 210 * 0.6  # G
        overlay[cloud_pixels, 2] = overlay[cloud_pixels, 2] * 0.4 + 255 * 0.6  # B
        
        # Clear regions: slight warm tint
        clear_pixels = ~cloud_pixels
        overlay[clear_pixels, 0] = overlay[clear_pixels, 0] * 0.7 + 80 * 0.3
        overlay[clear_pixels, 1] = overlay[clear_pixels, 1] * 0.7 + 60 * 0.3
        overlay[clear_pixels, 2] = overlay[clear_pixels, 2] * 0.7 + 40 * 0.3
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    
    def _array_to_base64(self, array: np.ndarray) -> str:
        """Convert a 2D numpy array (mask) to base64 PNG."""
        img = Image.fromarray((array * 255).astype(np.uint8), mode='L')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _pil_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 PNG."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
