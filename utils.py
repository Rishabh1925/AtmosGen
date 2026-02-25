import matplotlib.pyplot as plt

def visualize_frame(frame):

    frame = frame.detach().cpu().permute(1,2,0)

    plt.imshow(frame)
    plt.axis("off")
    plt.show()