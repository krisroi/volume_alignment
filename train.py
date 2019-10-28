import torch
import torch.optim as optim
import numpy as np
import math
import csv

# Folder dependent imports
import lib.network as network
import lib.affine as A
from lib.HDF5Image import HDF5Image
from lib.patch_volume import create_patches
from lib.ncc_loss import NCC
import lib.utils as ut
from visualize_res import loss_plot


def train_data(patched_data, epoch, epochs, lr, batch_size, net, criterion, optimizer, set_idx):

    fixed_patches = patched_data[:, 0, :].unsqueeze(1)
    moving_patches = patched_data[:, 1, :].unsqueeze(1)

    patch_loss = np.zeros(fixed_patches.shape[0])

    for idx in range(fixed_patches.shape[0]):

        optimizer.zero_grad()  # Zeroing the gradients

        # Perform forward pass on moving pathces
        if ((idx + batch_size) > fixed_patches.shape[0]):
            predicted_theta = net(moving_patches[(idx - batch_size):idx, :])

        elif ((idx - batch_size) < 0):
            predicted_theta = net(moving_patches[idx:(idx + batch_size), :])

        else:
            predicted_theta = net(moving_patches[(idx - int(batch_size / 2)):(idx + int(batch_size / 2)), :])

        predicted_deform = A.affine_transform(moving_patches[idx, :].unsqueeze(0), predicted_theta)

        loss = criterion(fixed_patches[idx, :].unsqueeze(0),
                         predicted_deform)

        loss.backward()
        patch_loss[idx] = loss.item()

        optimizer.step()

        if idx % 20 == 0:

            with open('output/txtfiles/loss2810191545', mode='a') as loss_file:
                loss_writer = csv.writer(loss_file, delimiter=',')
                loss_writer.writerow([idx, patch_loss[idx], (epoch + 1), epochs])

            print('====> Epoch: {}/{} \t Patch: {}/{} \t Datapart: {}/3 \t Patch loss: {}'
                  .format(epoch + 1, epochs, idx, fixed_patches.shape[0], set_idx + 1, np.round(patch_loss[idx], 4)))

    return patch_loss


def train_network(PROJ_ROOT, patient_group, patient,
                  fix_set, mov_set, epochs, lr, batch_size, patch_size,
                  stride, fixvol_no, movvol_no):

    with open('output/txtfiles/loss2810191545.csv', 'w') as new_file:
        fieldnames = ['patch_num', 'patch_loss', 'epoch', 'num_epochs', 'lr=' + str(lr),
                      'batch_size=' + str(batch_size), 'patch_size=' + str(patch_size), 'stride=' + str(stride)]
        writer = csv.DictWriter(new_file, fieldnames=fieldnames)
        writer.writeheader()

    net = network.Net()
    net.train()

    criterion = NCC()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    temp_loss = np.zeros(len(fix_set))
    epoch_loss = np.zeros(epochs)

    for epoch in range(epochs):

        for set_idx in range(len(fix_set)):

            print('Loading next set of volume data ...')
            vol_data = HDF5Image(PROJ_ROOT, patient_group, patient, fix_set[set_idx], mov_set[set_idx], fixvol_no, movvol_no)
            vol_data.normalize()

            print('Patching loaded data ...')
            patched_data = create_patches(vol_data.data, patch_size, stride)

            print('Training on the loaded data ...')
            training_loss = train_data(patched_data,
                                       epoch,
                                       epochs,
                                       lr,
                                       batch_size,
                                       net.train(),
                                       criterion,
                                       optimizer,
                                       set_idx
                                       )
            temp_loss[set_idx] = np.mean(training_loss)

        epoch_loss[epoch] = np.mean(temp_loss)
        print(epoch_loss[epoch])
    return epoch_loss


if __name__ == '__main__':

    # Defining filepath
    PROJ_ROOT = '/users/kristofferroise/project'
    patient_group = 'patient_data_proc/gr5_STolav5to8'
    patient = 'p7_3d'
    fixvol_no = 'vol01'
    movvol_no = 'vol01'

    fix_set = ['J249J70E_proc.h5',
               'J249J70E_proc.h5',
               'J249J70E_proc.h5']

    mov_set = ['J249J70I_proc.h5',
               'J249J70K_proc.h5',
               'J249J70G_proc.h5']

    # Creating known-transformation moving data
    theta_trans = torch.FloatTensor([[[0.98, 0, 0, -0.02],
                                      [0, 1, 0, 0.02],
                                      [0, 0, 1, -0.02]]])

    theta_idt = torch.FloatTensor([[[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]]])

    #=======================PARAMETERS==========================#
    lr = 1e-4  # learning rate
    epochs = 10  # number of epochs
    batch_size = 16
    patch_size = 30
    stride = 30
    train = True
    #===========================================================#
    if train:
        total_loss = train_network(PROJ_ROOT, patient_group, patient,
                                   fix_set, mov_set, epochs, lr, batch_size, patch_size,
                                   stride, fixvol_no, movvol_no)

        print(total_loss)
