import matplotlib.pyplot as plt
import torch
import math


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


def show_single(data, shape, title):
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

    fig.suptitle(title)

    plt.show()


def feature_maps(data, title):
    if data.shape[1] == 1:
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(20, 10))
        s1 = 1
        s2 = 1
    elif data.shape[1] == 8:
        fig, ax = plt.subplots(2, 4, squeeze=False, figsize=(20, 10))
        s1 = 2
        s2 = 4
    elif data.shape[1] == 16:
        fig, ax = plt.subplots(4, 4, squeeze=False, figsize=(20, 10))
        s1 = 4
        s2 = 4
    elif data.shape[1] == 32:
        fig, ax = plt.subplots(4, 8, squeeze=False, figsize=(20, 10))
        s1 = 4
        s2 = 8

    count = 0
    for k in range(s1):
        for j in range(s2):
            middle_slice = data[0, count, data.shape[2] - int((data.shape[2] / 3))]
            ax[k, j].imshow(middle_slice, origin='left', cmap='gray')
            ax[k, j].title.set_text('kernel ' + str(count))
            count += 1

    plt.suptitle(title)

    plt.show()
