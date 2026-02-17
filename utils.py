import torch
import matplotlib.pyplot as plt

def visualize_sequence(sequence):

    sequence = sequence.detach().cpu()

    fig, axes = plt.subplots(1, sequence.size(0), figsize=(15,4)) 

    for i in range(sequence.size(0)):
        img = sequence[i].permute(1,2,0)
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.show()