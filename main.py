import torch
import torch.optim as optim

from core_model.config import Config
from core_model.data.loader import get_dataloader
from core_model.models.unet import UNet
from core_model.models.diffusion import Diffusion
from core_model.trainer import Trainer


def main():

    config = Config()

    loader = get_dataloader(config, "core_model/data/satellite_images")

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