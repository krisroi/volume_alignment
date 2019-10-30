import torch
import torch.optim as optim
import numpy as np
import math
import csv
from datetime import datetime

# Folder dependent imports
from lib.network import Net
import lib.affine as A
from lib.HDF5Image import HDF5Image
from lib.patch_volume import create_patches
from lib.ncc_loss import NCC
import lib.utils as ut


def train_data(patched_data, epoch, epochs, lr, batch_size, net, criterion,
               optimizer, set_idx, num_sets, device):

    fixed_patches = patched_data[:, 0, :].unsqueeze(1).to(device)
    moving_patches = patched_data[:, 1, :].unsqueeze(1).to(device)

    patch_loss = torch.zeros(fixed_patches.shape[0], device=device)

    for idx in range(fixed_patches.shape[0]):

        optimizer.zero_grad()  # Zeroing the gradients

        # Perform forward pass on moving pathces
        if ((idx + batch_size) > fixed_patches.shape[0]):
            predicted_theta = net(moving_patches[(idx - batch_size):idx, :])

        elif ((idx - batch_size) < 0):
            predicted_theta = net(moving_patches[idx:(idx + batch_size), :])

        else:
            predicted_theta = net(moving_patches[(idx - int(batch_size / 2)):(idx + int(batch_size / 2)), :])

        predicted_deform = A.affine_transform(moving_patches[idx, :].unsqueeze(0).to(device),
                                              predicted_theta)

        loss = criterion(fixed_patches[idx, :].unsqueeze(0),
                         predicted_deform)

        loss.backward()
        patch_loss[idx] = loss.item()

        optimizer.step()

        if idx % 30 == 0:
            print('====> Epoch: {}/{} \t Datapart: {}/{} \t Patch: {}/{}'  # \t Remaining time: Calculating ...'
                  .format(epoch + 1, epochs, set_idx + 1, num_sets, idx, fixed_patches.shape[0]))

    return patch_loss


def train_network(PROJ_ROOT, patient_group, patient,
                  fix_set, mov_set, epochs, lr, batch_size, patch_size,
                  stride, fixvol_no, movvol_no, loss_path, num_sets, device):
    '''with open(loss_path, 'w') as els:
        fieldnames = ['epoch', 'epoch_loss', 'lr=' + str(lr), 'batch_size=' + str(batch_size),
                      'patch_size=' + str(patch_size), 'stride=' + str(stride),
                      'number_of_datasets=' + str(num_sets), 'device=' + str(device)]
        epoch_writer = csv.DictWriter(els, fieldnames=fieldnames)
        epoch_writer.writeheader()'''

    net = Net().to(device)
    net.train()

    criterion = NCC().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    temp_loss = torch.zeros(len(fix_set)).to(device)
    epoch_loss = torch.zeros(epochs).to(device)

    for epoch in range(epochs):

        # Print statements only valid for CUDA
        #print('Max memory allocated: ', (((torch.cuda.max_memory_allocated(device=device)/1024)/1024)/1024))
        #print('Max memory cached: ', (((torch.cuda.max_memory_cached(device=device)/1024)/1024)/1024))

        #print('Current memory allocated: ', (((torch.cuda.memory_allocated(device=device)/1024)/1024)/1024))
        #print('Current memory cached: ', (((torch.cuda.memory_cached(device=device)/1024)/1024)/1024))

        for set_idx in range(len(fix_set)):

            print('Loading next set of volume data ...')
            vol_data = HDF5Image(PROJ_ROOT, patient_group, patient, fix_set[set_idx],
                                 mov_set[set_idx], fixvol_no, movvol_no)
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
                                       set_idx,
                                       num_sets,
                                       device
                                       )
            temp_loss[set_idx] = torch.mean(training_loss)

        epoch_loss[epoch] = torch.mean(temp_loss)

        '''with open(loss_path, mode='a') as epoch_file:
            epoch_writer = csv.writer(epoch_file, delimiter=',')
            epoch_writer.writerow([(epoch + 1), epoch_loss[epoch]])'''

    return epoch_loss


if __name__ == '__main__':

    # Defining filepath
    PROJ_ROOT = '/users/kristofferroise/'
    patient_group = 'project'
    patient = 'patient_data_proc'
    fixvol_no = 'vol01'
    movvol_no = 'vol01'

    fix_set = ['DataStOlavs19to28/p24_3115009/J65BP2AQ_proc.h5',
               'DataStOlavs19to28/p23_3115008/J65BP22K_proc.h5',
               'DataStOlavs19to28/p19_3115004/J65BP12E_proc.h5',
               'DataStOlavs9to18/p10_3191405/J44J71A4_proc.h5',
               'DataStOlavs9to18/p14_3191409/J44J72A0_proc.h5']

    mov_set = ['DataStOlavs19to28/p24_3115009/J65BP2B0_proc.h5',
               'DataStOlavs19to28/p23_3115008/J65BP22M_proc.h5',
               'DataStOlavs19to28/p19_3115004/J65BP12G_proc.h5',
               'DataStOlavs9to18/p10_3191405/J44J71AG_proc.h5',
               'DataStOlavs9to18/p14_3191409/J44J729S_proc.h5']

    '''fix_set = ['gr4_STolav1to4/p3122154/J1ECATIE_proc.h5',
               'gr4_STolav1to4/p3122155/J1ECATQ2_proc.h5',
               'gr4_STolav1to4/p3122156/J1ECAU2M_proc.h5',
               'gr5_STolav5to8/p5_3d/J249J6IS_proc.h5',
               'gr5_STolav5to8/p6_3d/J249J6QQ_proc.h5',
               'gr5_STolav5to8/p7_3d/J249J70K_proc.h5',
               'gr5_STolav5to8/p8_3d/J249J79K_proc.h5'
    ]'''

    '''mov_set = ['gr4_STolav1to4/p3122154/J1ECATI8_proc.h5',
               'gr4_STolav1to4/p3122155/J1ECATQA_proc.h5',
               'gr4_STolav1to4/p3122156/J1ECAU2S_proc.h5',
               'gr5_STolav5to8/p5_3d/J249J6J2_proc.h5',
               'gr5_STolav5to8/p6_3d/J249J6QU_proc.h5',
               'gr5_STolav5to8/p7_3d/J249J70E_proc.h5',
               'gr5_STolav5to8/p8_3d/J249J79S_proc.h5'
    ]'''

    # Creating known-transformation moving data
    theta_trans = torch.FloatTensor([[[0.98, 0, 0, -0.02],
                                      [0, 1, 0, 0.02],
                                      [0, 0, 1, -0.02]]])

    theta_idt = torch.FloatTensor([[[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]]])

    #=======================PARAMETERS==========================#
    lr = 1e-4  # learning rate
    epochs = 1  # number of epochs
    batch_size = 16
    patch_size = 30
    stride = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #===========================================================#

    #=======================SAVING DATA=========================#
    train = True
    now = datetime.now()
    date = now.strftime('%d%m%Y')
    time = now.strftime('%X')
    loss_path = 'output/txtfiles/loss_{}_epochs_{}_{}'.format(epochs, date, time)
    #===========================================================#
    if train:
        start_time = datetime.now()
        total_loss = train_network(PROJ_ROOT, patient_group, patient,
                                   fix_set, mov_set, epochs, lr, batch_size, patch_size,
                                   stride, fixvol_no, movvol_no, loss_path, len(mov_set), device)
        print('Total time elapsed: ', datetime.now() - start_time)
