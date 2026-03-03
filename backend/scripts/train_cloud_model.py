#!/usr/bin/env python3
"""
Train cloud segmentation model on processed GOES-18 satellite data.

Fine-tunes a pre-trained U-Net (EfficientNet-B0 encoder) for binary
cloud segmentation using brightness temperature images.

Usage:
    python backend/scripts/train_cloud_model.py

Prerequisites:
    Run process_goes_data.py first to create training data.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Dataset ──────────────────────────────────────────────────

class CloudDataset(Dataset):
    """Dataset for cloud segmentation training."""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        self.image_files = sorted([
            f for f in os.listdir(images_dir) 
            if f.endswith('.png') or f.endswith('.jpg')
        ])
        
        # Verify matching masks exist
        valid_files = []
        for f in self.image_files:
            if os.path.exists(os.path.join(masks_dir, f)):
                valid_files.append(f)
        self.image_files = valid_files
        
        print(f"  Found {len(self.image_files)} image-mask pairs")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.image_files[idx])
        
        # Load image (already 3-channel from processing)
        image = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
        
        # Load mask (single channel, binary)
        mask = np.array(Image.open(mask_path).convert('L')).astype(np.float32) / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensors: (H, W, C)  (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        # Normalize with ImageNet stats (for pre-trained encoder)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image, mask


# ─── Loss Function ────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """Combined Binary Cross-Entropy + Dice loss for segmentation."""
    
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
    
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        
        # Dice loss
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        
        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


# ─── Metrics ──────────────────────────────────────────────────

def compute_iou(pred_mask, true_mask, threshold=0.5):
    """Compute Intersection over Union."""
    pred = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred * true_mask).sum()
    union = pred.sum() + true_mask.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def compute_dice(pred_mask, true_mask, threshold=0.5):
    """Compute Dice coefficient."""
    pred = (torch.sigmoid(pred_mask) > threshold).float()
    smooth = 1e-6
    intersection = (pred * true_mask).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + true_mask.sum() + smooth)


def compute_accuracy(pred_mask, true_mask, threshold=0.5):
    """Compute pixel-wise accuracy."""
    pred = (torch.sigmoid(pred_mask) > threshold).float()
    correct = (pred == true_mask).float().sum()
    total = true_mask.numel()
    return (correct / total).item()


# ─── Training ─────────────────────────────────────────────────

def get_augmentations():
    """Get training augmentations using albumentations."""
    try:
        import albumentations as A
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        ])
    except ImportError:
        print("  Warning: albumentations not installed, skipping augmentations")
        return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    total_iou = 0
    n_batches = 0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_iou += compute_iou(logits.detach(), masks)
        n_batches += 1
    
    return total_loss / n_batches, total_iou / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model, return metrics."""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_acc = 0
    n_batches = 0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        logits = model(images)
        loss = criterion(logits, masks)
        
        total_loss += loss.item()
        total_iou += compute_iou(logits, masks)
        total_dice += compute_dice(logits, masks).item()
        total_acc += compute_accuracy(logits, masks)
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'iou': total_iou / n_batches,
        'dice': total_dice / n_batches,
        'accuracy': total_acc / n_batches,
    }


def main():
    print("=" * 60)
    print("AtmosGen — Cloud Segmentation Model Training")
    print("=" * 60)
    
    # ── Config ──
    EPOCHS = 30
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    PATIENCE = 7  # Early stopping patience
    
    # ── Paths ──
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_images = os.path.join(data_dir, 'train', 'images')
    train_masks = os.path.join(data_dir, 'train', 'masks')
    val_images = os.path.join(data_dir, 'val', 'images')
    val_masks = os.path.join(data_dir, 'val', 'masks')
    
    # Verify data exists
    if not os.path.exists(train_images):
        print(f"\nERROR: Training data not found at {train_images}")
        print("Run: python backend/scripts/process_goes_data.py first")
        sys.exit(1)
    
    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\nDevice: {device}")
    
    # ── Datasets ──
    print("\nLoading datasets...")
    augmentations = get_augmentations()
    
    train_dataset = CloudDataset(train_images, train_masks, transform=augmentations)
    val_dataset = CloudDataset(val_images, val_masks, transform=None)
    
    if len(train_dataset) == 0:
        print("ERROR: No training data found!")
        sys.exit(1)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # ── Model ──
    print("\nInitializing model...")
    
    import segmentation_models_pytorch as smp
    
    model = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',  # Pre-trained on ImageNet
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {param_count:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Encoder: EfficientNet-B0 (ImageNet pre-trained)")
    print(f"  Architecture: U-Net")
    
    # ── Training Setup ──
    criterion = DiceBCELoss(dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # ── Training Loop ──
    print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience: {PATIENCE})")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val IoU':>10} | {'Val Dice':>10} | {'Val Acc':>10} | {'Time':>6}")
    print("-" * 80)
    
    best_iou = 0.0
    patience_counter = 0
    training_history = []
    
    start_total = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        start_epoch = time.time()
        
        # Train
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['iou'])
        
        epoch_time = time.time() - start_epoch
        
        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_metrics['loss']:>10.4f} | "
              f"{val_metrics['iou']:>10.4f} | {val_metrics['dice']:>10.4f} | "
              f"{val_metrics['accuracy']:>10.4f} | {epoch_time:>5.1f}s")
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_iou': val_metrics['iou'],
            'val_dice': val_metrics['dice'],
            'val_accuracy': val_metrics['accuracy'],
        })
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            patience_counter = 0
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_iou': best_iou,
                'val_metrics': val_metrics,
                'training_history': training_history,
                'model_config': {
                    'encoder_name': 'efficientnet-b0',
                    'encoder_weights': 'imagenet',
                    'in_channels': 3,
                    'classes': 1,
                    'architecture': 'Unet',
                }
            }
            
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'cloud_seg_best.pth'))
            print(f"       ↳ Saved best model (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break
    
    total_time = time.time() - start_total
    
    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best Val IoU: {best_iou:.4f}")
    print(f"  Best Val Dice: {training_history[np.argmax([h['val_iou'] for h in training_history])]['val_dice']:.4f}")
    print(f"  Best Val Accuracy: {training_history[np.argmax([h['val_iou'] for h in training_history])]['val_accuracy']:.4f}")
    print(f"\n  Checkpoint: checkpoints/cloud_seg_best.pth")
    print(f"\n  Next step: python backend/main.py  (start the API server)")
    
    # Save a final checkpoint too
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'cloud_seg_v1.pth'))


if __name__ == '__main__':
    main()
