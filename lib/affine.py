import torch
import torch.nn.functional as F
from lib.HDF5Image import HDF5Image
from lib.patch_volume import create_patches
import numpy as np
import lib.utils as ut
from lib.ncc_loss import NCC


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
        Return size: [B, D, H, W, 3]
    """

    B, C, D, H, W = size  # Extract dimensions of the input

    theta = theta.expand(B, 3, 4)  # expand to the number of batches you need

    grid = F.affine_grid(theta, size=(B, C, D, H, W))
    grid = grid.view(B, D, H, W, 3)

    return grid


def affine_transform(data, theta, patch=True):
    """ Performs an affine transform of some input data with a transformation matrix theta
    Args:
        data: input the volume data that is to be transformed
        theta: transformation matrix to do the transformation
        patch: TRUE if the data to transform is patch volumes (default)
               FALSE if the data to transform is full volume
    Returns:
        Transformed input data that is transformed with the transformation matrix.
        shape: [B, C, D, H, W]
    """
    if patch:
        # Extracting only the moving image in the 'data'-variable
        moving = data[:, 0, :]
        moving = moving.unsqueeze(1)  # Re-adding channel-dimension
        B, C, D, H, W = moving.shape  # Extracting the dimensions
    else:
        # Extracting only the moving image in the 'data'-variable
        moving = data[1, :]
        moving = moving.unsqueeze(0)  # Re-adding channel-dimension
        moving = moving.unsqueeze(0)  # Adding batch dimension (1)
        B, C, D, H, W = moving.shape  # Extracting the dimensions

    grid = affine_grid_3d((B, C, D, H, W), theta)
    warped = F.grid_sample(moving, grid)
    #print('Warped image shape: ', warped.shape)

    return warped


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

    crit = NCC()

    vol_data = HDF5Image(PROJ_ROOT, patient_group, patient,
                         fixfile, movfile,
                         fixvol_no, movvol_no)

    input_batch = create_patches(vol_data.data, patch_size, stride)

    # Transformation matrix with slight rotation
    theta_trans = torch.FloatTensor([[[0.98, 0.02, 0, 0.02],
                                      [0.02, 1, 0, 0.02],
                                      [0, 0, 1, 0.02]]])

    # Identity matrix (No rotation)
    theta_idt = torch.FloatTensor([[[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]]])

    show_patches = True

    if show_patches:
        data = input_batch
        warped_image = affine_transform(data, theta_idt, patch=True)
        fixed_image = data[:, 1, :]  # This is moving data and NOT fixed for visual inspection only
        fixed_print_image = data[255, 1, :]
        ut.show_single(fixed_print_image.unsqueeze(0), fixed_print_image.unsqueeze(0).shape)  # Printing the original 255th patch in the moving image
        ut.show_single(warped_image[255, :], warped_image[255, :].shape)  # Printing the warped 255th patch in the moving image
        loss = crit(fixed_image, warped_image, patch=True)
        print('Patch image loss: ', loss)

    else:
        data = vol_data.data
        warped_image = affine_transform(data, theta_trans, patch=False)
        fixed_image = data[1, :]
        ut.show_single(fixed_image.unsqueeze(0), fixed_image.unsqueeze(0).shape)  # Printing the original moving image
        ut.show_single(warped_image[0, :],
                       warped_image[0, :].shape)  # Printing the warped moving image
        loss = crit(fixed_image, warped_image, patch=False)
        print('Full image loss: ', loss.item())
