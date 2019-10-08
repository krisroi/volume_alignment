import torch
import torch.nn.functional as F
from HDF5Image import load_hdf5
from patch_volume import create_patches
import numpy as np
import cv2


def affine_grid_3d(theta, size):
    B, C, D, H, W = size

    theta = theta.expand(B, 3, 4)  # expand to the number of batches you need

    grid = F.affine_grid(theta, size=(B, 1, D, H, W))
    grid = grid.view(B, D, H, W, 3)

    return grid


def affine_transform(moving, theta):

    moving = moving[1]
    N, D, H, W = moving.shape
    print(" N: " + str(N) + ". D: " + str(D) + ". H: " + str(H) + ". W: " + str(W))

    moving = moving.unsqueeze(1)

    grid = affine_grid_3d(theta, (N, 1, D, H, W))

    print(grid.shape)
    predicted = F.grid_sample(moving, grid)

    return predicted


if __name__ == "__main__":
    fixed_file = 'J249J70K_proc.h5'
    moving_file = 'J249J70M_proc.h5'

    data, shape = load_hdf5(fixed_file, moving_file)
    image = data.unsqueeze(1)

    #theta = torch.FloatTensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])
    theta = torch.FloatTensor([[[1, 0, 0, 0.08],
                                [0, 1, 0, -0.07],
                                [0, 0, 1, 0.12]]])

    warped_image = affine_transform(image, theta)
    warped_image = np.array(warped_image, dtype=np.uint8)
    print("Warped image shape: " + str(warped_image.shape))

    cv2.imshow('x', warped_image[0, 0, int(warped_image.shape[2] / 2), :, :])
    cv2.imshow('y', warped_image[0, 0, :, int(warped_image.shape[3] / 2), :])
    cv2.imshow('z', warped_image[0, 0, :, :, int(warped_image.shape[4] / 2)])
    #cv2.imwrite('/users/kristofferroise/trans_x.png', warped_image[0, 0, int(warped_image.shape[2] / 2), :, :])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
