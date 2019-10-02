from lib.HDF5Image import load_hdf5
import lib.utils as ut
from lib.patch_volume import create_patches
import lib.network as network
import torch
import torch.optim as optim
import torch.nn as nn

fixed_file = 'J249J70K_proc.h5'
moving_file = 'J249J70M_proc.h5'

data, shape = load_hdf5(fixed_file, moving_file)

data_norm = ut.normalize_pixels(data)

stride = 20
patch_size = 20

input_batch = create_patches(data_norm, patch_size, stride)

print(input_batch[0, 1:].shape)
#ut.show_single(input_batch[234, 1:], input_batch[234, 1:].shape)

print(input_batch[234:235, 1:].shape)

net = network.Net()
print(net)

# input = torch.randn(1, 1, 20, 20, 20)  # Random input
input = input_batch[237:240, 1:]  # Real data input
out = net(input)
print(out)

#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
