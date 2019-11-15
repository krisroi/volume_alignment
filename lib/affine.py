import torch
import torch.nn.functional as F


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
