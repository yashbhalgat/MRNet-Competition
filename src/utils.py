import numpy as np
import matplotlib.pyplot as plt

def plot_data(data):
    width = data.shape[0]//5 +1
    fig, axs = plt.subplots(5, width)
    for i in range(data.shape[0]):
        axs[i//width, i%width].imshow(data[i,:,:])
        axs[i//width, i%width].set_yticklabels([])
        axs[i//width, i%width].set_xticklabels([])
    plt.show()
