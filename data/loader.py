from torch.utils.data import DataLoader
from .dataset import SatelliteDataset

def get_dataloader(config, data_path):

    dataset = SatelliteDataset(
        data_path,
        config.SEQUENCE_LENGTH,
        config.PREDICT_STEPS,
        config.IMAGE_SIZE
    )

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )