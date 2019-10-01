import lib.get_patch_pos as util
import time
import torch
from torch.nn import functional as F


def create_patches(data, patch_size, stride):

    stride = stride
    patch_size = patch_size

    mod = len(data[0, :]) % patch_size

    if mod != 0:
        pad_size = patch_size - mod
        padded = F.pad(data, (pad_size, 0, pad_size, 0, pad_size, 0))
        data = padded

    data_size = data.shape
    N = data_size[0]

    start = time.time() * 1
    for i in range(N - 1):
        print(i)
        flat_idx = util.calculatePatchIdx3D(1, patch_size * torch.ones(3), data_size[1:], stride * torch.ones(3))
        flat_idx_select = torch.zeros(flat_idx.size())

        for patch_idx in range(1, flat_idx.size()[0]):
            patch_pos = util.idx2pos_4D(flat_idx[patch_idx], data_size[1:])

            fixed_patch = data.data[i,
                                    patch_pos[1]:patch_pos[1] + patch_size,
                                    patch_pos[2]:patch_pos[2] + patch_size,
                                    patch_pos[3]:patch_pos[3] + patch_size]
            moving_patch = data.data[i + 1,
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size,
                                     patch_pos[3]:patch_pos[3] + patch_size]

            if (torch.sum(moving_patch) + torch.sum(fixed_patch) != 0):
                flat_idx_select[patch_idx] = 1
            # end if
        # end for
        flat_idx_select = flat_idx_select.bool()
        flat_idx = torch.masked_select(flat_idx, flat_idx_select)

        input_batch = torch.zeros(flat_idx.shape[0], 2, patch_size, patch_size, patch_size)

        for slices in range(flat_idx.shape[0]):
            patch_pos = util.idx2pos_4D(flat_idx[slices], data_size[1:])
            input_batch[slices, 0] = data.data[i,
                                               patch_pos[1]:patch_pos[1] + patch_size,
                                               patch_pos[2]:patch_pos[2] + patch_size,
                                               patch_pos[3]:patch_pos[3] + patch_size]
            input_batch[slices, 1] = data.data[i + 1,
                                               patch_pos[1]:patch_pos[1] + patch_size,
                                               patch_pos[2]:patch_pos[2] + patch_size,
                                               patch_pos[3]:patch_pos[3] + patch_size]
        # end for
    # end for
    stop = time.time() * 1
    print(stop - start)

    return input_batch
