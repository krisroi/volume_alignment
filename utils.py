import matplotlib.pyplot as plt
import torch


def normalize_pixels(data):
    data = data / 255.0
    tensor = torch.empty(data.shape, dtype=torch.float64)
    data = data.clone().detach()
    return data


def show_dual(data, num_volumes, shape):
    fig, ax = plt.subplots(2, 3, squeeze=False, figsize=(20, 10))

    for i in range(num_volumes):

        middle_slice = data[i - 1, int(shape[1] / 2)]
        ax[i - 1, 0].imshow(middle_slice, origin='left', cmap='gray')

        middle_slice = data[i - 1, :, int(shape[2] / 2)]
        ax[i - 1, 1].imshow(middle_slice, origin='middle', cmap='gray')

        middle_slice = data[i - 1, :, :, int(shape[3] / 2)]
        ax[i - 1, 2].imshow(middle_slice, origin='right', cmap='gray')

        for j in range(len(data.shape) - 1):
            ax[i - 1, j].set_xlim([0, len(middle_slice)])
            ax[i - 1, j].set_ylim([len(middle_slice), 0])

    plt.show()


def show_single(data, shape):
    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(20, 10))

    middle_slice = data[0, int(shape[1] / 2)]
    ax[0, 0].imshow(middle_slice, origin='left', cmap='gray')

    middle_slice = data[0, :, int(shape[2] / 2)]
    ax[0, 1].imshow(middle_slice, origin='middle', cmap='gray')

    middle_slice = data[0, :, :, int(shape[3] / 2)]
    ax[0, 2].imshow(middle_slice, origin='right', cmap='gray')

    for j in range(3):
        ax[0, j].set_xlim([0, len(middle_slice)])
        ax[0, j].set_ylim([len(middle_slice), 0])

    plt.show()
