from lib.HDF5Image import load_hdf5
import lib.utils as ut
from lib.patch_volume import create_patches

fixed_file = 'J249J70K_proc.h5'
moving_file = 'J249J70M_proc.h5'

data, shape = load_hdf5(fixed_file, moving_file)

data_norm = ut.normalize_pixels(data)

stride = 20
patch_size = 20

input_batch = create_patches(data_norm, patch_size, stride)

ut.show_single(input_batch[31, :], input_batch[565, :].shape)
