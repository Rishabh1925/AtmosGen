from config import Config
from data.loader import get_dataloader
import torch
from models.unet import UNet
from models.diffusion import Diffusion
import torch.nn.functional as F
import torch.optim as optim



def main():

    # Initialize config
    config = Config()

    # Data loader
    data_path = "data/satellite_images"
    loader = get_dataloader(config, data_path)

    # Initialize model
    model = UNet(sequence_length=config.SEQUENCE_LENGTH).to(config.DEVICE)

    # Initialize diffusion
    diffusion = Diffusion()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


    model.train()

    for inputs, targets in loader:

        inputs = inputs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)

        # Use first future frame for diffusion training
        target_frame = targets[:, 0]

        # Sample diffusion timesteps
        t = diffusion.sample_timestamps(inputs.size(0)).to(config.DEVICE)

        # Add noise
        noisy_images, noise = diffusion.add_noise(target_frame, t)

        # Predict noise
        predicted_noise = model(inputs, t)

        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Training Loss:", loss.item())

        break


if __name__ == "__main__":
    main()

