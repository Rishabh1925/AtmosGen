from torch.utils.data import DataLoader
from .dataset import SatelliteDataset

def get_dataloader(config, data_path):

    dataset = SatelliteDataset(
        root_dir = data_path,
        sequence_length=config.SEQUENCE_LENGTH,
        predict_steps=config.PREDICT_STEPS,
        image_size=config.IMAGE_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    return loader