import torch

class Config:

    #Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Data
    IMAGE_SIZE = 128
    SEQUENCE_LENGTH = 8
    PREDICT_STEPS = 4
    BATCH_SIZE = 4

    #Training
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-4