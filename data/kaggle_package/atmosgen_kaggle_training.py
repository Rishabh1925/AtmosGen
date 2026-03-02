
# AtmosGen Weather Forecasting - Kaggle Training Notebook
# Fine-tune Stable Diffusion for weather forecasting

# Cell 1: Install dependencies
!pip install diffusers transformers accelerate xformers -q
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

# Cell 2: Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
import os
import json
import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt

print("✅ Libraries imported successfully!")

# Cell 3: Extract and load data
print("📦 Extracting training data...")

# Extract the uploaded dataset
with zipfile.ZipFile('/kaggle/input/atmosgen-weather-training-data/atmosgen_training_data.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/data')

# Load dataset info
with open('/kaggle/working/data/dataset_info.json', 'r') as f:
    dataset_info = json.load(f)

print(f"✅ Dataset loaded: {dataset_info['total_samples']} samples")
print(f"📅 Created: {dataset_info['created_at']}")

# Cell 4: Custom Dataset Class
class WeatherForecastDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load dataset info
        with open(self.data_dir / 'dataset_info.json', 'r') as f:
            self.dataset_info = json.load(f)
        
        self.samples = self.dataset_info['samples']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load input sequence (3 images)
        input_images = []
        for input_file in sample['input_files']:
            img_path = self.data_dir / 'inputs' / input_file
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            input_images.append(img)
        
        # Load target forecast
        target_path = self.data_dir / 'targets' / sample['target_file']
        target_img = Image.open(target_path).convert('RGB')
        if self.transform:
            target_img = self.transform(target_img)
        
        return {
            'input_sequence': torch.stack(input_images),  # [3, C, H, W]
            'target': target_img,  # [C, H, W]
            'description': sample['description']
        }

# Image transforms
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Create dataset and dataloader
dataset = WeatherForecastDataset('/kaggle/working/data', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

print(f"✅ Dataset ready: {len(dataset)} samples")

# Cell 5: Load pre-trained Stable Diffusion
print("🤖 Loading pre-trained Stable Diffusion model...")

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load components
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Move to device
text_encoder = text_encoder.to(device)
unet = unet.to(device)

print(f"✅ Model loaded on {device}")

# Cell 6: Prepare for fine-tuning
# Freeze text encoder (we only fine-tune the UNet)
text_encoder.requires_grad_(False)

# Setup optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5, weight_decay=1e-2)

# Weather-related prompts for conditioning
weather_prompts = [
    "satellite weather image showing clouds and atmospheric patterns",
    "meteorological satellite view of weather systems",
    "weather forecast satellite imagery with cloud formations",
    "atmospheric conditions from satellite perspective",
    "weather patterns visible from space"
]

def encode_prompt(prompt):
    """Encode text prompt for conditioning"""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    
    return text_embeddings

print("✅ Fine-tuning setup complete")

# Cell 7: Training loop
print("🚀 Starting fine-tuning...")

num_epochs = 3
global_step = 0

# Pre-encode weather prompts
encoded_prompts = [encode_prompt(prompt) for prompt in weather_prompts]

for epoch in range(num_epochs):
    unet.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Get batch data
        input_sequence = batch['input_sequence']  # [B, 3, C, H, W]
        target = batch['target']  # [B, C, H, W]
        
        batch_size = target.shape[0]
        
        # Move to device
        target = target.to(device)
        
        # Random timestep for diffusion training
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
        
        # Add noise to target (diffusion forward process)
        noise = torch.randn_like(target)
        noisy_target = scheduler.add_noise(target, noise, timesteps)
        
        # Random weather prompt for conditioning
        prompt_idx = torch.randint(0, len(encoded_prompts), (batch_size,))
        encoder_hidden_states = torch.stack([encoded_prompts[i] for i in prompt_idx])
        
        # Predict noise
        with torch.cuda.amp.autocast():
            noise_pred = unet(noisy_target, timesteps, encoder_hidden_states).sample
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        global_step += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Log every 10 steps
        if global_step % 10 == 0:
            avg_loss = epoch_loss / (batch_idx + 1)
            print(f"Step {global_step}, Avg Loss: {avg_loss:.4f}")
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")

print("🎉 Fine-tuning completed!")

# Cell 8: Save fine-tuned model
print("💾 Saving fine-tuned model...")

# Create output directory
output_dir = "/kaggle/working/atmosgen_finetuned"
os.makedirs(output_dir, exist_ok=True)

# Save UNet (the fine-tuned component)
unet.save_pretrained(f"{output_dir}/unet")

# Save other components (unchanged but needed for inference)
text_encoder.save_pretrained(f"{output_dir}/text_encoder")
tokenizer.save_pretrained(f"{output_dir}/tokenizer")

# Save scheduler config
scheduler.save_pretrained(f"{output_dir}/scheduler")

# Create model config
model_config = {
    "model_type": "stable_diffusion_weather",
    "base_model": model_id,
    "fine_tuned_on": "weather_forecasting",
    "training_samples": len(dataset),
    "epochs": num_epochs,
    "final_loss": avg_epoch_loss
}

with open(f"{output_dir}/model_config.json", 'w') as f:
    json.dump(model_config, f, indent=2)

print(f"✅ Model saved to: {output_dir}")

# Cell 9: Create downloadable package
print("📦 Creating downloadable package...")

import shutil

# Create zip file for download
shutil.make_archive("/kaggle/working/atmosgen_checkpoint", 'zip', output_dir)

print("✅ Download 'atmosgen_checkpoint.zip' from the output section!")
print("🎯 Model is ready for integration with AtmosGen!")

# Cell 10: Test the fine-tuned model
print("🧪 Testing fine-tuned model...")

# Create pipeline with fine-tuned UNet
from diffusers import StableDiffusionPipeline

# Load the fine-tuned pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    unet=unet,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    torch_dtype=torch.float16
)
pipe = pipe.to(device)

# Generate a test weather forecast
test_prompt = "satellite weather image showing storm development and cloud formations"
test_image = pipe(test_prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

# Display result
plt.figure(figsize=(8, 8))
plt.imshow(test_image)
plt.title("Fine-tuned Weather Forecast Sample")
plt.axis('off')
plt.show()

print("🎉 Fine-tuning complete! Model ready for weather forecasting!")
