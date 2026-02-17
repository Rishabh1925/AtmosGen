import torch
import cv2
import numpy as np

def preprocess_image(image_path, image_size):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))

    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2,0,1))

    return torch.tensor(image)
