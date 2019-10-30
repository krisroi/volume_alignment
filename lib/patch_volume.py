import time
import torch
from torch.nn import functional as F


#======================================================================================#
# Yang, X., Kwitt, R., Styner, M., Niethammer, M., 2017.
# Quicksilver: Fast predictive imageregistration - a deep learning approach.
# arXiv:1703.10908.

def calculateIdx1D(length, patch_length, step):
    one_dim_pos = torch.arange(0, length - patch_length + 1, step)
    if (length - patch_length) % step != 0:
        one_dim_pos = torch.cat((one_dim_pos, torch.ones(1) * (length - patch_length)))
    return one_dim_pos


def idx2pos(idx, image_size):
    """
    Given a flattened idx, return the position in the 3D image space.
    Args:
        idx (int):                  Index into flattened 3D volume
        image_size(list of 3 int):  Size of 3D volume
    """
    assert (len(image_size) == 3)

    pos_x = idx / (image_size[1] * image_size[2])
    idx_yz = idx % (image_size[1] * image_size[2])
    pos_y = idx_yz / image_size[2]
    pos_z = idx_yz % image_size[2]
    return torch.LongTensor([pos_x, pos_y, pos_z])


def pos2idx(pos, image_size):
    """
    Given a position in the 3D image space, return a flattened idx.
    Args:
        pos (list of 3 int):        Position in 3D volume
        image_size (list of 3 int): Size of 3D volume
    """
    assert (len(pos) == 3)
    assert (len(image_size) == 3)

    return (pos[0] * image_size[1] * image_size[2]) + (pos[1] * image_size[2]) + pos[2]


# given a flatterned idx for a 4D data (n_images * 3D image), return the position in the 4D space
def idx2pos_4D(idx, image_size):
    image_slice = idx / (image_size[0] * image_size[1] * image_size[2])
    single_image_idx = idx % (image_size[0] * image_size[1] * image_size[2])
    single_image_pos = idx2pos(single_image_idx, image_size)
    return torch.cat((image_slice * torch.ones(1).long(), single_image_pos))


# calculate the idx of the patches for 3D dataset (n_images * 3D image)
def calculatePatchIdx3D(num_image, patch_size, image_size, step_size):
    # calculate the idx for 1 3D image
    pos_idx = [calculateIdx1D(image_size[i], patch_size[i], step_size[i]).long() for i in range(0, 3)]
    pos_idx_flat = torch.zeros(pos_idx[0].size()[0] * pos_idx[1].size()[0] * pos_idx[2].size()[0]).long()
    flat_idx = 0
    pos_3d = torch.zeros(3).long()
    for x_pos in range(0, pos_idx[0].size()[0]):
        for y_pos in range(0, pos_idx[1].size()[0]):
            for z_pos in range(0, pos_idx[2].size()[0]):
                pos_3d[0] = pos_idx[0][x_pos]
                pos_3d[1] = pos_idx[1][y_pos]
                pos_3d[2] = pos_idx[2][z_pos]
                pos_idx_flat[flat_idx] = pos2idx(pos_3d, image_size)
                flat_idx = flat_idx + 1

    pos_idx_flat_all = pos_idx_flat.long()

    # calculate the idx across all 3D images in the dataset
    for i in range(1, num_image):
        pos_idx_flat_all = torch.cat(
            (pos_idx_flat_all, pos_idx_flat.long() + i * (image_size[0] * image_size[1] * image_size[2])))

    return pos_idx_flat_all
#======================================================================================#


def create_patches(data, patch_size, stride):
    """ Patches the volumetric input data, using the patch_size and the stride.
    Args:
        data: volumetric data, both moving and fixed
        patch_size: desired shape of each patch (patch_size x patch_size x patch_size)
        stride: stride length of the patches, defines the overlap. If stride = patch_size, there is no overlap.
    Returns:
        Patched volumetric data, defined by the patch size and stride.
        shape: [num_patches, num_channels, patch_size, patch_size, patch_size]
    """

    # Returns zero if the dimensions of the volume is zero-dividable with the patch_size
    mod = len(data[0, :]) % patch_size

    # Zero-pads the volume if the mod statement does not return zero
    if mod != 0:
        pad_size = patch_size - mod
        padded = F.pad(data, (pad_size, 0, pad_size, 0, pad_size, 0))
        data = padded

    data_size = data.shape
    N = data_size[0]

    # USELESS FOR-LOOP, consider removing
    for i in range(N - 1):
        flat_idx = calculatePatchIdx3D(1, patch_size * torch.ones(3), data_size[1:], stride * torch.ones(3))
        flat_idx_select = torch.zeros(flat_idx.size())

        for patch_idx in range(1, flat_idx.size()[0]):
            patch_pos = idx2pos_4D(flat_idx[patch_idx], data_size[1:])

            fixed_patch = data.data[i,
                                    patch_pos[1]:patch_pos[1] + patch_size,
                                    patch_pos[2]:patch_pos[2] + patch_size,
                                    patch_pos[3]:patch_pos[3] + patch_size]
            moving_patch = data.data[i + 1,
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size,
                                     patch_pos[3]:patch_pos[3] + patch_size]

            fix_on = torch.ones(fixed_patch.shape)
            fix_off = torch.zeros(fixed_patch.shape)
            mov_on = torch.ones(moving_patch.shape)
            mov_off = torch.zeros(moving_patch.shape)

            # Checking where in the patches there is data and where it is not
            fixed_on = torch.where(fixed_patch != 0, fix_on, fix_off)
            moving_on = torch.where(moving_patch != 0, mov_on, mov_off)

            # Selecting only the patches that contain > 40% non-zero data
            if (torch.sum(fixed_on) >= (patch_size**3) * 0.4) & (torch.sum(moving_on) >= (patch_size**3) * 0.4):
                flat_idx_select[patch_idx] = 1

        # end for
        flat_idx_select = flat_idx_select.bool()
        flat_idx = torch.masked_select(flat_idx, flat_idx_select)

        patched_data = torch.zeros(flat_idx.shape[0], 2, patch_size, patch_size, patch_size)

        for slices in range(flat_idx.shape[0]):
            patch_pos = idx2pos_4D(flat_idx[slices], data_size[1:])
            patched_data[slices, 0] = data.data[i,
                                                patch_pos[1]:patch_pos[1] + patch_size,
                                                patch_pos[2]:patch_pos[2] + patch_size,
                                                patch_pos[3]:patch_pos[3] + patch_size]
            patched_data[slices, 1] = data.data[i + 1,
                                                patch_pos[1]:patch_pos[1] + patch_size,
                                                patch_pos[2]:patch_pos[2] + patch_size,
                                                patch_pos[3]:patch_pos[3] + patch_size]
        # end for
    # end for

    return patched_data
