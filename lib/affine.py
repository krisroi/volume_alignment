import torch
import torch.nn.functional as F
import numpy as np


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


def affine_transform(data, theta):
    """ Performs an affine transform of some input data with a transformation matrix theta
    Args:
        data: input the volume data that is to be transformed
        theta: transformation matrix to do the transformation
    Returns:
        Transformed input data that is transformed with the transformation matrix.
        shape: [B, C, D, H, W]
    """
    B, C, D, H, W = data.shape  # Extracting the dimensions

    grid_3d = affine_grid_3d((B, C, D, H, W), theta)
    warped_patches = F.grid_sample(data, grid_3d, padding_mode='border')  # alternatives: 'border', 'zeros', 'reflection'

    return warped_patches


if __name__ == "__main__":

    import utils as ut
    from ncc_loss import NCC
    from HDF5Image import HDF5Image
    from patch_volume import create_patches

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

    patched_data = create_patches(vol_data.data, patch_size, stride)

    # Transformation matrix with slight rotation
    theta_trans = torch.FloatTensor([[[0.98, 0.02, 0, 0.02],
                                      [0.02, 1, 0, 0.02],
                                      [0, 0, 1, 0.02]]])

    # Identity matrix (No rotation)
    theta_idt = torch.FloatTensor([[[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]]])

    data = patched_data[:, 1, :].unsqueeze(1)
    warped_image = affine_transform(data, theta_trans)
    fixed_image = patched_data[:, 1, :].unsqueeze(1)  # This is moving data and NOT fixed for visual inspection only
    fixed_print_image = patched_data[254, 1, :]
    ut.show_single(fixed_print_image.unsqueeze(0), fixed_print_image.unsqueeze(0).shape, title='Fixed patches')  # Printing the original 255th patch in the moving image
    ut.show_single(warped_image[254, :], warped_image[255, :].shape, title='Warped patches')  # Printing the warped 255th patch in the moving image
    loss = NCC()(fixed_image, warped_image)
    print('Patch image loss: ', loss.item())
