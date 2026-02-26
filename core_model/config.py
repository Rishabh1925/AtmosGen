import torch

class Config:

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = 128
    SEQUENCE_LENGTH = 8
    PREDICT_STEPS = 4

    BATCH_SIZE = 4
    NUM_WORKERS = 2

    LEARNING_RATE = 1e-4
    EPOCHS = 5

    TIMESTEPS = 500