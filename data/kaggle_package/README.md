# AtmosGen Weather Forecasting Training Data

## Overview
This dataset contains synthetic weather sequences for training the AtmosGen weather forecasting model.

## Contents
- `training_data/inputs/` - Input satellite image sequences (3 images per sequence)
- `training_data/targets/` - Target forecast images (1 per sequence)
- `dataset_info.json` - Dataset metadata and sample information
- `atmosgen_kaggle_training.py` - Complete training script for Kaggle

## Usage
1. Upload this dataset to Kaggle
2. Create a new notebook with GPU enabled
3. Copy the code from `atmosgen_kaggle_training.py`
4. Run the training cells in order
5. Download the fine-tuned model checkpoint

## Model Architecture
- Base: Stable Diffusion v1.5
- Fine-tuning: UNet component only
- Task: Weather sequence forecasting
- Input: 3 satellite images → Output: 1 forecast image

## Training Details
- Samples: 40 weather sequences (8 patterns × 5 variations)
- Epochs: 3-5 (adjustable)
- Batch size: 2 (GPU memory optimized)
- Learning rate: 1e-5
- Expected training time: 1-2 hours on Kaggle GPU

## Expected Results
After fine-tuning, the model should generate realistic weather patterns including:
- Cloud formation and movement
- Storm development
- Clear to cloudy transitions
- Hurricane/cyclone structures

Ready for weather AI training! 
