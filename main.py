import torch
import torch.optim as optim

from config import Config
from data.loader import get_dataloader
from models.unet import UNet
from models.diffusion import Diffusion
from trainer import Trainer


def main():

    config = Config()

    loader = get_dataloader(config, "data/satellite_images")

    model = UNet().to(config.DEVICE)
    diffusion = Diffusion(config.TIMESTEPS)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    trainer = Trainer(model, diffusion, optimizer, config)

    for epoch in range(config.EPOCHS):

        loss = trainer.train_epoch(loader)

        psnr_score, ssim_score = trainer.evaluate(loader)

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {loss:.4f} | "
            f"PSNR: {psnr_score:.2f} | "
            f"SSIM: {ssim_score:.4f}"
        )

        trainer.save_checkpoint(epoch+1)


if __name__ == "__main__":
    main()


