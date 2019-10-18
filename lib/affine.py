import torch
import torch.nn.functional as F
from HDF5Image import HDF5Image
from patch_volume import create_patches
import numpy as np
import cv2


def affine_grid_3d(size, theta):
    """ Defines an affine grid in 3 dimensions.
    Args:
        size: Inputs the size of the input batch in the form of
            B = batch size
            C = number of channels
            D, H, W = dimensions of the input volume in depth, height and width
        theta: Inputs a transformation matrix
    Returns:
        A 3d affine grid that is used lated for transformation
        Return size: [batch size, D, H, W, 3]
    """

    B, C, D, H, W = size

    theta = theta.expand(B, 3, 4)  # expand to the number of batches you need

    grid = F.affine_grid(theta, size=(B, 1, D, H, W))
    grid = grid.view(B, D, H, W, 3)
    print('Grid shape: ', grid.shape)

    return grid


def affine_transform(data, theta):
    """ Performs an affine transform of some input data with a transformation matrix theta
    Args:
        data: input the volume data that is to be transformed
        theta: transformation matrix to do the transformation
    Returns:
        Transformed input data that is transformed with the transformation matrix.
        shape: []
    """

    B, C, D, H, W = data.shape
    print(" B: " + str(B) + ". D: " + str(D) + ". H: " + str(H) + ". W: " + str(W))

    #moving = data.unsqueeze(1)

    grid = affine_grid_3d((B, 1, D, H, W), theta)

    transformed = F.grid_sample(moving, grid)
    print('Transformed image shape: ', transformed.shape)

    return transformed


if __name__ == "__main__":
    PROJ_ROOT = '/users/kristofferroise/project'
    patient_group = 'patient_data/gr5_STolav5to8'
    patient = 'p7_3d'
    fixfile = 'J249J70K_proc.h5'
    movfile = 'J249J70M_proc.h5'
    fixvol_no = 'vol01'
    movvol_no = 'vol02'

    patch_size = 20
    stride = 20

    show_patches = False

    vol_data = HDF5Image(PROJ_ROOT, patient_group, patient,
                         fixfile, movfile,
                         fixvol_no, movvol_no)

    input_batch = create_patches(vol_data.data, patch_size, stride)

    if show_patches:
        moving = input_batch
    else:
        moving = vol_data.data
        moving = moving.unsqueeze(1)

    theta = torch.FloatTensor([[[0.7, -0.7, 0, 0],
                                [0.7, 0.7, 0, 0],
                                [0, 0, 1, 0]]])

    warped_image = affine_transform(moving, theta)
    warped_image = np.array(warped_image, dtype=np.uint8)
    original_image = np.array(moving, dtype=np.uint8)
    print("Warped image shape: ", warped_image.shape)
    print("Original image shape: ", original_image.shape)

    if show_patches:
        cv2.imshow('Original x', original_image[530, 1, int(original_image.shape[2] / 2), :, :])
        cv2.imshow('Warped x', warped_image[530, 1, int(warped_image.shape[2] / 2), :, :])
    else:
        cv2.imshow('Original x', original_image[0, 0, int(original_image.shape[2] / 2), :, :])
        cv2.imshow('Warped x', warped_image[0, 0, int(warped_image.shape[2] / 2), :, :])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
