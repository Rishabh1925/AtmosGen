import os
import torch
from torch.utils.data import Dataset
from .transforms import preprocess_image

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, sequence_length, predict_steps, image_size):

        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.predict_steps = predict_steps
        self.image_size = image_size

        self.images = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.images) - self.sequence_length - self.predict_steps
    
    def __getitem__(self, idx):

        input_sequence = []
        target_sequence = []

        for i in range(self.sequence_length):
            img_path = os.path.join(self.root_dir, self.images[idx + i])
            input_sequence.append(preprocess_image(img_path, self.image_size))

        for j in range(self.predict_steps):
            img_path = os.path.join(
                self.root_dir,
                self.images[idx + self.sequence_length + j]
            )
            target_sequence.append(preprocess_image(img_path, self.image_size))

        input_sequence = torch.stack(input_sequence)
        target_sequence = torch.stack(target_sequence)

        return input_sequence, target_sequence