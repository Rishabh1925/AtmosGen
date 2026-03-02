#!/usr/bin/env python3
"""
Proper weather forecasting model with realistic evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, structural_similarity
import cv2

class ImprovedWeatherNet(nn.Module):
    """Improved weather forecasting model with skip connections and attention."""
    
    def __init__(self):
        super().__init__()
        
        # Encoder with skip connections
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        
        # Decoder with skip connections
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv9 = nn.Conv2d(512, 256, 3, padding=1)  # 512 = 256 + 256 (skip)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv11 = nn.Conv2d(256, 128, 3, padding=1)  # 256 = 128 + 128 (skip)
        self.conv12 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, 3, padding=1)   # 128 = 64 + 64 (skip)
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.final_conv = nn.Conv2d(64, 3, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        p1 = self.pool1(x1)
        
        x2 = self.relu(self.conv3(p1))
        x2 = self.relu(self.conv4(x2))
        p2 = self.pool2(x2)
        
        x3 = self.relu(self.conv5(p2))
        x3 = self.relu(self.conv6(x3))
        p3 = self.pool3(x3)
        
        # Bottleneck
        x4 = self.relu(self.conv7(p3))
        x4 = self.relu(self.conv8(x4))
        
        # Decoder with skip connections
        up1 = self.upconv1(x4)
        merge1 = torch.cat([up1, x3], dim=1)
        x5 = self.relu(self.conv9(merge1))
        x5 = self.relu(self.conv10(x5))
        
        up2 = self.upconv2(x5)
        merge2 = torch.cat([up2, x2], dim=1)
        x6 = self.relu(self.conv11(merge2))
        x6 = self.relu(self.conv12(x6))
        
        up3 = self.upconv3(x6)
        merge3 = torch.cat([up3, x1], dim=1)
        x7 = self.relu(self.conv13(merge3))
        x7 = self.relu(self.conv14(x7))
        
        output = self.sigmoid(self.final_conv(x7))
        return output

def calculate_weather_metrics(pred_img, target_img):
    """Calculate weather-specific evaluation metrics."""
    
    # Convert to numpy arrays
    pred_np = pred_img.cpu().numpy()
    target_np = target_img.cpu().numpy()
    
    # Ensure proper shape (H, W, C)
    if pred_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        pred_np = np.transpose(pred_np, (1, 2, 0))
    if target_np.shape[0] == 3:
        target_np = np.transpose(target_np, (1, 2, 0))
    
    # Convert to grayscale for some metrics
    pred_gray = cv2.cvtColor((pred_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor((target_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    metrics = {}
    
    # Mean Squared Error
    metrics['mse'] = mean_squared_error(target_np.flatten(), pred_np.flatten())
    
    # Structural Similarity Index
    metrics['ssim'] = structural_similarity(target_gray, pred_gray)
    
    # Peak Signal-to-Noise Ratio
    mse_val = np.mean((pred_np - target_np) ** 2)
    if mse_val == 0:
        metrics['psnr'] = float('inf')
    else:
        metrics['psnr'] = 20 * np.log10(1.0 / np.sqrt(mse_val))
    
    # Cloud coverage similarity (using intensity)
    pred_intensity = np.mean(pred_gray)
    target_intensity = np.mean(target_gray)
    metrics['cloud_coverage_error'] = abs(pred_intensity - target_intensity) / 255.0
    
    # Edge preservation (weather patterns)
    pred_edges = cv2.Canny(pred_gray, 50, 150)
    target_edges = cv2.Canny(target_gray, 50, 150)
    edge_similarity = np.sum(pred_edges & target_edges) / np.sum(target_edges | pred_edges)
    metrics['edge_preservation'] = edge_similarity
    
    return metrics

def create_realistic_weather_sequence():
    """Create a more realistic weather sequence for testing."""
    
    # This would normally load real weather data
    # For now, create synthetic but realistic patterns
    
    base_image = np.random.rand(512, 512, 3) * 0.3 + 0.4  # Base atmosphere
    
    # Add cloud patterns
    x, y = np.meshgrid(np.linspace(0, 10, 512), np.linspace(0, 10, 512))
    clouds = np.sin(x) * np.cos(y) * 0.3 + 0.5
    
    # Apply clouds to all channels
    for c in range(3):
        base_image[:, :, c] *= clouds
    
    # Create temporal evolution (next frame)
    next_frame = base_image.copy()
    # Simulate cloud movement
    shifted_clouds = np.roll(clouds, 20, axis=1)  # Move clouds
    for c in range(3):
        next_frame[:, :, c] *= shifted_clouds
    
    return base_image, next_frame

def evaluate_weather_model():
    """Evaluate the weather model with proper metrics."""
    
    print("Weather Model Evaluation")
    print("=" * 40)
    
    # Check if we have a trained model
    checkpoint_path = "../checkpoints/atmosgen_simple_v1.pth"
    
    if os.path.exists(checkpoint_path):
        print("Loading existing trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the simple model for comparison
        from demo_model import SimpleWeatherNet as OldModel
        old_model = OldModel().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        old_model.load_state_dict(checkpoint['model_state_dict'])
        old_model.eval()
        
        print("Model loaded successfully")
        
        # Test with synthetic realistic data
        print("\nGenerating realistic test data...")
        input_weather, target_weather = create_realistic_weather_sequence()
        
        # Convert to tensors
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        input_tensor = transform(input_weather.astype(np.float32)).unsqueeze(0).to(device)
        target_tensor = transform(target_weather.astype(np.float32)).unsqueeze(0).to(device)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Target shape: {target_tensor.shape}")
        
        # Generate prediction
        print("\nGenerating prediction...")
        with torch.no_grad():
            prediction = old_model(input_tensor)
        
        print(f"Prediction shape: {prediction.shape}")
        
        # Calculate metrics
        print("\nCalculating evaluation metrics...")
        metrics = calculate_weather_metrics(prediction[0], target_tensor[0])
        
        print("\nWeather Prediction Metrics:")
        print(f"  Mean Squared Error: {metrics['mse']:.6f}")
        print(f"  Structural Similarity: {metrics['ssim']:.4f}")
        print(f"  Peak SNR: {metrics['psnr']:.2f} dB")
        print(f"  Cloud Coverage Error: {metrics['cloud_coverage_error']:.4f}")
        print(f"  Edge Preservation: {metrics['edge_preservation']:.4f}")
        
        # Interpret results
        print("\nModel Performance Assessment:")
        if metrics['ssim'] > 0.7:
            print("  SSIM: GOOD - Structural similarity is acceptable")
        elif metrics['ssim'] > 0.5:
            print("  SSIM: FAIR - Some structural similarity preserved")
        else:
            print("  SSIM: POOR - Low structural similarity")
        
        if metrics['psnr'] > 20:
            print("  PSNR: GOOD - Low noise in predictions")
        elif metrics['psnr'] > 15:
            print("  PSNR: FAIR - Moderate noise levels")
        else:
            print("  PSNR: POOR - High noise in predictions")
        
        if metrics['cloud_coverage_error'] < 0.1:
            print("  Cloud Coverage: GOOD - Accurate cloud density")
        elif metrics['cloud_coverage_error'] < 0.2:
            print("  Cloud Coverage: FAIR - Reasonable cloud density")
        else:
            print("  Cloud Coverage: POOR - Inaccurate cloud density")
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Top row: Input, Target, Prediction
        axes[0, 0].imshow(input_weather)
        axes[0, 0].set_title("Input Weather")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(target_weather)
        axes[0, 1].set_title("Target Weather (Ground Truth)")
        axes[0, 1].axis('off')
        
        pred_img = prediction[0].cpu().permute(1, 2, 0).numpy()
        axes[0, 2].imshow(pred_img)
        axes[0, 2].set_title("Model Prediction")
        axes[0, 2].axis('off')
        
        # Bottom row: Difference maps
        diff_target = np.abs(target_weather - input_weather)
        axes[1, 0].imshow(diff_target)
        axes[1, 0].set_title("Expected Change")
        axes[1, 0].axis('off')
        
        diff_pred = np.abs(pred_img - input_weather)
        axes[1, 1].imshow(diff_pred)
        axes[1, 1].set_title("Predicted Change")
        axes[1, 1].axis('off')
        
        error_map = np.abs(pred_img - target_weather)
        axes[1, 2].imshow(error_map)
        axes[1, 2].set_title("Prediction Error")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig("weather_evaluation_results.png", dpi=150, bbox_inches='tight')
        print("\nEvaluation results saved: weather_evaluation_results.png")
        
        # Overall assessment
        overall_score = (metrics['ssim'] + (metrics['psnr']/30) + (1-metrics['cloud_coverage_error']) + metrics['edge_preservation']) / 4
        
        print(f"\nOverall Model Score: {overall_score:.3f} / 1.000")
        
        if overall_score > 0.7:
            print("ASSESSMENT: Model shows good weather prediction capability")
        elif overall_score > 0.5:
            print("ASSESSMENT: Model shows moderate weather prediction capability")
        else:
            print("ASSESSMENT: Model needs significant improvement for weather prediction")
        
    else:
        print("No trained model found. Please train a model first.")
        return False
    
    return True

if __name__ == "__main__":
    evaluate_weather_model()