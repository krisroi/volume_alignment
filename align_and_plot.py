import matplotlib
matplotlib.use('tkagg')

import torch
import pandas as pd
import matplotlib.pyplot as plt

from lib.affine import affine_transform
from lib.HDF5Image import HDF5Image


def plot_fixed_moving(fixed_volume, moving_volume, warped_volume, copper_alpha, gray_alpha):

    fig, ax = plt.subplots(2, 3, squeeze=False, figsize=(20, 6))

    fixed_xslice = fixed_volume[0, int(fixed_volume.shape[1] / 2)]
    warped_xslice = warped_volume[0, int(warped_volume.shape[1] / 2)]
    moving_xslice = moving_volume[0, int(moving_volume.shape[1] / 2)]

    fixed_yslice = fixed_volume[0, :, int(fixed_volume.shape[2] / 2)]
    warped_yslice = warped_volume[0, :, int(warped_volume.shape[2] / 2)]
    moving_yslice = moving_volume[0, :, int(moving_volume.shape[1] / 2)]

    fixed_zslice = fixed_volume[0, :, :, int(fixed_volume.shape[3] / 2)]
    warped_zslice = warped_volume[0, :, :, int(warped_volume.shape[3] / 2)]
    moving_zslice = moving_volume[0, :, :, int(moving_volume.shape[1] / 2)]

    ax[0, 0].imshow(fixed_xslice, origin='left', cmap='copper', alpha=copper_alpha)
    ax[0, 0].imshow(moving_xslice, origin='left', cmap='gray', alpha=gray_alpha)
    ax[0, 0].set_xlim([0, fixed_volume.shape[1]])
    ax[0, 0].set_ylim([fixed_volume.shape[2], 0])

    ax[0, 1].imshow(fixed_yslice, origin='middle', cmap='copper', alpha=copper_alpha)
    ax[0, 1].imshow(moving_yslice, origin='middle', cmap='gray', alpha=gray_alpha)
    ax[0, 1].set_xlim([0, fixed_volume.shape[1]])
    ax[0, 1].set_ylim([fixed_volume.shape[3], 0])

    ax[0, 2].imshow(fixed_zslice, origin='right', cmap='copper', alpha=copper_alpha)
    ax[0, 2].imshow(moving_zslice, origin='right', cmap='gray', alpha=gray_alpha)
    ax[0, 2].set_xlim([0, fixed_volume.shape[2]])
    ax[0, 2].set_ylim([fixed_volume.shape[3], 0])

    ax[1, 0].imshow(fixed_xslice, origin='left', cmap='copper', alpha=copper_alpha)
    ax[1, 0].imshow(warped_xslice, origin='left', cmap='gray', alpha=gray_alpha)
    ax[1, 0].set_xlim([0, fixed_volume.shape[1]])
    ax[1, 0].set_ylim([fixed_volume.shape[2], 0])

    ax[1, 1].imshow(fixed_yslice, origin='middle', cmap='copper', alpha=copper_alpha)
    ax[1, 1].imshow(warped_yslice, origin='middle', cmap='gray', alpha=gray_alpha)
    ax[1, 1].set_xlim([0, fixed_volume.shape[1]])
    ax[1, 1].set_ylim([fixed_volume.shape[3], 0])

    ax[1, 2].imshow(fixed_zslice, origin='right', cmap='copper', alpha=copper_alpha)
    ax[1, 2].imshow(warped_zslice, origin='right', cmap='gray', alpha=gray_alpha)
    ax[1, 2].set_xlim([0, fixed_volume.shape[1]])
    ax[1, 2].set_ylim([fixed_volume.shape[3], 0])


def align(theta_file, path_to_h5files copper_alpha, gray_alpha):

    global_theta = torch.zeros([12])

    with open(theta_file, 'r') as f:
        for i, theta in enumerate(f.read().split()):
            if theta != '1' and theta != '0':
                global_theta[i] = float(theta)

    global_theta = global_theta.view(-1, 3, 4)

    fixed_image = 'DataStOlavs19to28/p22_3115007/J65BP1R0_proc.h5'
    moving_image = 'DataStOlavs19to28/p22_3115007/J65BP1R2_proc.h5'
    fix_vol = '01'
    mov_vol = '12'

    vol_data = HDF5Image(path_to_h5files, fixed_image, moving_image, fix_vol, mov_vol)

    fixed_volume = vol_data.data[0, :].unsqueeze(0)
    moving_volume = vol_data.data[1, :].unsqueeze(0).unsqueeze(1)

    predicted_deformation = affine_transform(moving_volume, global_theta)

    plot_fixed_moving(fixed_volume, moving_volume.squeeze(1), predicted_deformation.squeeze(1), copper_alpha, gray_alpha)


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')

    theta_file = 'output/txtfiles/res.txt'
    path_to_h5files = '/mnt/EncryptedFastData/krisroi/patient_data_proc/'

    copper_alpha = 1
    gray_alpha = 1

    align(theta_file, path_to_h5files, copper_alpha, gray_alpha)