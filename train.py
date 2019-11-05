import torch
import torch.optim as optim
import numpy as np
import math
import csv
from datetime import datetime
from sklearn.utils import shuffle

# Folder dependent imports
from lib.network import Net
import lib.affine as A
from lib.HDF5Image import HDF5Image
from lib.patch_volume import create_patches
from lib.ncc_loss import NCC
import lib.utils as ut
from lib.dataloader import Dataset


def train_data(patched_data, epoch, epochs, lr, batch_size, net, criterion,
               optimizer, set_idx, num_sets, device):

    fixed_patches = patched_data[:, 0, :].unsqueeze(1).to(device)
    moving_patches = patched_data[:, 1, :].unsqueeze(1).to(device)

    patch_loss = torch.zeros(fixed_patches.shape[0], device=device)

    for idx in range(fixed_patches.shape[0]):

        optimizer.zero_grad()  # Zeroing the gradients

        if ((idx + batch_size) > fixed_patches.shape[0]):
            idx_arr = torch.LongTensor(range((idx - batch_size), idx))

        elif ((idx - batch_size) < 0):
            idx_arr = torch.LongTensor(range(idx, (idx + batch_size)))

        else:
            idx_arr = torch.LongTensor(range((idx - int(batch_size / 2)), (idx + int(batch_size / 2))))

        predicted_theta = net(moving_patches[idx_arr, :])

        predicted_deform = A.affine_transform(moving_patches[idx_arr, :],
                                              predicted_theta)

        loss = criterion(fixed_patches[idx_arr, :],
                         predicted_deform, reduction='mean')

        loss.backward()
        patch_loss[idx] = loss.item()

        optimizer.step()

        if idx % 30 == 0:
            print('====> Epoch: {}/{} \t Datapart: {}/{} \t Patch: {}/{}'  # \t Remaining time: Calculating ...'
                  .format(epoch + 1, epochs, set_idx + 1, num_sets, idx, fixed_patches.shape[0]))

        cur_state_dict = net.state_dict()

    return patch_loss, cur_state_dict


def train_network(path_to_files, fix_set, mov_set, epochs, lr, batch_size, patch_size,
                  stride, fix_vols, mov_vols, loss_path, num_sets, device, model_name, voxelsize):
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

        for set_idx in range(num_sets):

            print('Loading next set of volume data ...')
            vol_data = HDF5Image(path_to_files, fix_set[set_idx], mov_set[set_idx],
                                 fix_vols[set_idx], mov_vols[set_idx])
            vol_data.normalize()

            print('Patching loaded data ...')
            patched_data, loc = create_patches(vol_data.data, patch_size, stride, device, voxelsize)
            patch_location = 'output/txtfiles/patch_location.csv'

            with open(patch_location, 'w') as lctn:
                fieldnames = ['x_pos', 'y_pos', 'z_pos']
                field_writer = csv.DictWriter(lctn, fieldnames=fieldnames)
                field_writer.writeheader()
                lctn_writer = csv.writer(lctn, delimiter=',')
                lctn_writer.writerows(loc.cpu().numpy().round(5))

            print('Training on the loaded data ...')
            training_loss, cur_state_dict = train_data(patched_data,
                                                       epoch,
                                                       epochs,
                                                       lr,
                                                       batch_size,
                                                       net.train(),
                                                       criterion,
                                                       optimizer,
                                                       set_idx,
                                                       num_sets,
                                                       device,
                                                       )

            temp_loss[set_idx] = torch.mean(training_loss)

        '''model_info = {'patch_size': patch_size,
                      'state_dict': cur_state_dict}
        print('Saving model ... ')
        torch.save(model_info, model_name)'''

        epoch_loss[epoch] = torch.mean(temp_loss)

        '''with open(loss_path, mode='a') as epoch_file:
            epoch_writer = csv.writer(epoch_file, delimiter=',')
            epoch_writer.writerow([(epoch + 1), epoch_loss[epoch]])'''

    return epoch_loss


if __name__ == '__main__':

    #=======================PARAMETERS==========================#
    lr = 1e-4  # learning rate
    epochs = 1  # number of epochs
    use_sets = 26  # number of datasets that we wish to use. Max of 26
    shuffle_sets = True
    batch_size = 16
    patch_size = 30
    stride = 30
    voxelsize = 7.0000003e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #===========================================================#

    #=======================SAVING DATA=========================#
    train = True
    now = datetime.now()
    date = now.strftime('%d%m%Y')
    time = now.strftime('%H%M%S')
    model_name = 'output/models/model_{}_{}.pt'.format(date, time)
    loss_path = 'output/txtfiles/loss_{}_epochs_{}_{}.csv'.format(epochs, date, time)
    #===========================================================#

    # Defining filepath to .h5 files
    path_to_files = '/users/kristofferroise/project/patient_data_proc/'

    # Filepath to dataset_information file
    path_to_info = '/users/kristofferroise/project/Diverse/'
    info_filename = 'dataset_information.csv'

    dataset = Dataset(path_to_info, info_filename)

    fix_set = dataset.fix_files
    mov_set = dataset.mov_files
    fix_vols = dataset.fix_vols
    mov_vols = dataset.mov_vols

    # Shuffling datasets
    if shuffle_sets:
        fix_set, mov_set, fix_vols, mov_vols = shuffle(fix_set, mov_set, fix_vols, mov_vols)

    if train:
        start_time = datetime.now()
        total_loss = train_network(path_to_files, fix_set, mov_set, epochs, lr, batch_size, patch_size,
                                   stride, fix_vols, mov_vols, loss_path, use_sets, device, model_name, voxelsize)
        print('Total time elapsed: ', datetime.now() - start_time)
