import torch
import torch.nn.functional as F
from .sampling import sample
from .metrics import psnr, ssim
import os


class Trainer:

    def __init__(self, model, diffusion, optimizer, config):

        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.config = config

        os.makedirs("checkpoints", exist_ok=True)

    def train_epoch(self, loader):

        self.model.train()
        total_loss = 0

        for inputs, targets in loader:

            inputs = inputs.to(self.config.DEVICE)
            targets = targets.to(self.config.DEVICE)

            target_frame = targets[:, 0]

            t = self.diffusion.sample_timesteps(inputs.size(0)).to(self.config.DEVICE)

            noisy_images, noise = self.diffusion.add_noise(target_frame, t)

            conditional_input = torch.cat(
                [inputs, noisy_images.unsqueeze(1)], dim=1
            )

            predicted_noise = self.model(conditional_input, t)

            loss = F.mse_loss(predicted_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader):

        self.model.eval()

        with torch.no_grad():

            inputs, targets = next(iter(loader))

            inputs = inputs.to(self.config.DEVICE)
            targets = targets.to(self.config.DEVICE)

            generated = sample(
                self.model, self.diffusion, inputs, self.config.DEVICE
            )

            target_frame = targets[:, 0]

            return psnr(generated, target_frame).item(), \
                   ssim(generated, target_frame).item()

    def save_checkpoint(self, epoch):

        torch.save(
            self.model.state_dict(),
            f"checkpoints/atmosgen_epoch_{epoch}.pth"
        )