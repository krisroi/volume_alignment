import torch
import torch.optim as optim
import lib.network as network
import lib.affine as A
from lib.HDF5Image import HDF5Image
from lib.patch_volume import create_patches
import numpy as np
from lib.ncc_loss import NCC
import lib.utils as ut


def train_network(fixed_image, warped_image, learning_rate, epochs):
    net = network.Net()
    net.train()

    criterion = NCC()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print('Fixed shape: {} \t Warped shape: {}'.format(fixed_image.shape, warped_image.shape))

    N = int(fixed_image.shape[0] / 32)

    for epoch in range(epochs):

        for idx in range(fixed_image.shape[0] - 64):

            optimizer.zero_grad()  # Zeroing the gradients

            # Perform forward pass on moving pathces
            predicted_theta = net(warped_image[idx:idx + 63, :])
            predicted_deform = A.affine_transform(warped_image, predicted_theta, patch=True)

            #print('Predicted deformation shape: ', predicted_deform.shape)
            #ut.show_single(predicted_deform[0, :].detach().numpy(), predicted_deform[0, :].shape)
            #ut.show_single(fixed[patch, :].unsqueeze(0), fixed.shape)

            loss = criterion(fixed_image[idx, :].unsqueeze(0),
                             predicted_deform[idx, :].unsqueeze(0), patch=True)
            loss.backward()
            loss_val = loss.item()

            optimizer.step()

            if idx % 50 == 0:
                print('====> Epoch: {} #\t Loss: {}'.format(epoch, loss_val))

    # return predicted_theta


if __name__ == '__main__':

    # Defining filepath
    PROJ_ROOT = '/users/kristofferroise/project'
    patient_group = 'patient_data/gr5_STolav5to8'
    patient = 'p7_3d'
    fixfile = 'J249J70K_proc.h5'
    movfile = 'J249J70M_proc.h5'
    fixvol_no = 'vol01'
    movvol_no = 'vol02'

    # Loading volumetric data
    vol_data = HDF5Image(PROJ_ROOT, patient_group, patient,
                         fixfile, movfile,
                         fixvol_no, movvol_no)

    patch_size = 20
    stride = 20

    patched_data = create_patches(vol_data.data, patch_size, stride)

    # Creating known-transformation moving data
    theta_trans = torch.FloatTensor([[[0.98, 0.02, 0, 0.02],
                                      [0.02, 1, 0, 0.02],
                                      [0, 0, 1, 0.02]]])

    theta_idt = torch.FloatTensor([[[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]]])

    warped_image = A.affine_transform(patched_data, theta_trans, patch=True)
    fixed_image = patched_data[:, 0, :]

    #=======================PARAMETERS==========================#
    lr = 0.001  # learning rate
    epochs = 1  # number of epochs
    #===========================================================#

    predicted_theta = train_network(fixed_image, warped_image, lr, epochs)
    print(predicted_theta)
