# AtmosGen Kaggle Training Guide
## Option A: Pre-trained Model + Fine-tuning for Weather

---

## Overview

We'll use a pre-trained Stable Diffusion model and fine-tune it for weather forecasting. This gives us:
- ✅ **Realistic image generation** from day 1
- ✅ **Fast training** (1-2 hours on Kaggle GPU)
- ✅ **Good results** without massive datasets
- ✅ **Professional-looking outputs** for demos

---

## Phase 1: Prepare Training Data (Local - 15 minutes)

### Step 1: Create Diverse Weather Dataset
We'll create a synthetic weather dataset with realistic variations:

```bash
# Run this locally to create training data
cd backend
python create_training_dataset.py
```

This will create:
- 200+ weather image pairs (input sequence → target forecast)
- Different weather conditions (clear, cloudy, stormy)
- Temporal variations (morning → afternoon, storm development)
- Geographic diversity (ocean, land, mixed)

### Step 2: Package for Kaggle Upload
```bash
# Create Kaggle dataset package
python package_for_kaggle.py
```

Creates: `atmosgen_training_data.zip` (ready for Kaggle upload)

---

## Phase 2: Kaggle Setup (Kaggle - 10 minutes)

### Step 1: Create New Kaggle Notebook
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Select "GPU P100" (free tier)
4. Title: "AtmosGen Weather Forecasting Training"

### Step 2: Upload Training Data
1. Create new dataset: [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Upload `atmosgen_training_data.zip`
3. Title: "AtmosGen Weather Training Data"
4. Make it public
5. Add to your notebook as data source

---

## Phase 3: Training Code (Kaggle - 1-2 hours)

### Step 1: Install Dependencies
```python
# Kaggle notebook cell 1
!pip install diffusers transformers accelerate xformers
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Load Pre-trained Model
```python
# Kaggle notebook cell 2
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import torch

# Load pre-trained Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

print("✅ Pre-trained model loaded successfully!")
```

### Step 3: Prepare Weather Data
```python
# Kaggle notebook cell 3
import os
import zipfile
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Extract training data
with zipfile.ZipFile('/kaggle/input/atmosgen-weather-training-data/atmosgen_training_data.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/data')

# Custom weather dataset
class WeatherDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sequences = self._load_sequences()
    
    def _load_sequences(self):
        # Load weather sequences (input → target pairs)
        sequences = []
        # Implementation will be in the generated script
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Return input sequence and target forecast
        pass

dataset = WeatherDataset('/kaggle/working/data')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"✅ Dataset loaded: {len(dataset)} training samples")
```

### Step 4: Fine-tuning Loop
```python
# Kaggle notebook cell 4
from diffusers import DDPMScheduler
import torch.nn.functional as F

# Setup training
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-5)
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Fine-tuning implementation
        # Will be detailed in the generated script
        pass
    
    print(f"✅ Epoch {epoch+1}/{num_epochs} completed")

print("🎉 Fine-tuning completed!")
```

### Step 5: Save Trained Model
```python
# Kaggle notebook cell 5
# Save the fine-tuned model
pipe.save_pretrained("/kaggle/working/atmosgen_finetuned")

# Create downloadable checkpoint
import shutil
shutil.make_archive("/kaggle/working/atmosgen_checkpoint", 'zip', "/kaggle/working/atmosgen_finetuned")

print("✅ Model saved! Download 'atmosgen_checkpoint.zip'")
```

---

## Phase 4: Integration (Local - 15 minutes)

### Step 1: Download and Install
1. Download `atmosgen_checkpoint.zip` from Kaggle
2. Extract to `checkpoints/atmosgen_finetuned/`
3. Update model service to use new checkpoint

### Step 2: Test Fine-tuned Model
```bash
cd backend
python test_finetuned_model.py
```

### Step 3: Update Frontend
The fine-tuned model will work with existing frontend - no changes needed!

---

## Expected Results

### Before Fine-tuning:
- Random-looking outputs
- No weather patterns
- Inconsistent quality

### After Fine-tuning:
- ✅ **Realistic weather patterns** (clouds, storms, clear skies)
- ✅ **Temporal consistency** (logical weather evolution)
- ✅ **Professional quality** suitable for demos
- ✅ **Fast inference** (5-10 seconds per forecast)

---

## Timeline

| Phase | Time | Location | Task |
|-------|------|----------|------|
| 1 | 15 min | Local | Create training data |
| 2 | 10 min | Kaggle | Setup notebook & upload data |
| 3 | 1-2 hours | Kaggle | Fine-tune model |
| 4 | 15 min | Local | Download & integrate |
| **Total** | **2-2.5 hours** | | **Complete fine-tuned model** |

---

## Next Steps

1. **Run the data preparation scripts** (I'll create them next)
2. **Follow the Kaggle training guide** step by step
3. **Download and integrate** the fine-tuned model
4. **Test with your NOAA data** - should see much better results!

Ready to start? I'll create the data preparation scripts now!