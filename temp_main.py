from lib.HDF5Image import HDF5Image
import lib.utils as ut
from lib.patch_volume import create_patches
import lib.network as network
import torch
import torch.optim as optim
import torch.nn as nn

PROJ_ROOT = '/users/kristofferroise/project'
patient_group = 'patient_data/gr5_STolav5to8'
patient = 'p7_3d'
fixfile = 'J249J70K_proc.h5'
movfile = 'J249J70M_proc.h5'
fixvol_no = 'vol01'
movvol_no = 'vol02'

image = HDF5Image(PROJ_ROOT, patient_group, patient,
                  fixfile, movfile,
                  fixvol_no, movvol_no)
print(image.data.shape)
fixed = image.data[0, :].unsqueeze(0)
moving = image.data[1:2, :]

print(fixed.shape)

#data_norm = ut.normalize_pixels(image.data)

stride = 20
patch_size = 20

input_batch = create_patches(image.data, patch_size, stride)

single = input_batch[250, 0:]
ut.show_single(single, single.shape)
