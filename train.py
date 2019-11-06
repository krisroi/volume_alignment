import torch
from torch.utils.data import Dataset, DataLoader
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
from lib.dataloader import GetDatasetInformation


class CreateDataset(Dataset):
    def __init__(self, fixed_patches, moving_patches):
        self.fixed_patches = fixed_patches
        self.moving_patches = moving_patches

    def __len__(self):
        return self.fixed_patches.shape[0]

    def __getitem__(self, idx):
        sample = {'fixed_patches': self.fixed_patches[idx, :],
                  'moving_patches': self.moving_patches[idx, :]}
        return sample


def validate(path_to_files, val_fix_set, val_mov_set, val_fix_vols, val_mov_vols, batch_size, net, criterion, device):

    for val_set_idx in range(len(val_fix_set)):

        validation_data = HDF5Image(path_to_files, val_fix_set[val_set_idx], val_mov_set[val_set_idx],
                                    val_fix_vols[val_set_idx], val_mov_vols[val_set_idx])
        validation_data.normalize()
        validation_data.to(device)

        patched_validation_data, _ = create_patches(validation_data.data, patch_size, stride, device, voxelsize)

        fixed_patches = patched_validation_data[:, 0, :].unsqueeze(1).to(device)
        moving_patches = patched_validation_data[:, 1, :].unsqueeze(1).to(device)

        validation_set = CreateDataset(fixed_patches, moving_patches)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        val_set_loss = torch.zeros(int(fixed_patches.shape[0] / batch_size) + 1, device=device)
        validation_loss = torch.zeros(num_validation_sets, device=device)

        for batch_idx, val_data in enumerate(validation_loader):

            batched_fixed_patches = val_data['fixed_patches']
            batched_moving_patches = val_data['moving_patches']

            predicted_theta = net(batched_moving_patches)

            predicted_deform = A.affine_transform(batched_moving_patches, predicted_theta)

            loss = criterion(batched_fixed_patches, predicted_deform, reduction='mean')

            val_set_loss[batch_idx] = loss.item()

            if batch_idx % 1 == 0:
                print('====> Validating ... \t Datapart: {}/{} \t Batch: {}/{}'  # \t Remaining time: Calculating ...'
                      .format(val_set_idx + 1, num_validation_sets, batch_idx + 1, int(fixed_patches.shape[0] / batch_size) + 1))

        validation_loss[val_set_idx] = torch.mean(val_set_loss)

    return torch.mean(validation_loss)


def train_data(patched_data, epoch, epochs, lr, batch_size, net, criterion,
               optimizer, set_idx, num_sets, device, num_validation_sets):

    fixed_patches = patched_data[:, 0, :].unsqueeze(1).to(device)
    moving_patches = patched_data[:, 1, :].unsqueeze(1).to(device)

    train_set = CreateDataset(fixed_patches, moving_patches)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    patch_loss = torch.zeros(int(fixed_patches.shape[0] / batch_size) + 1, device=device)

    for batch_idx, train_data in enumerate(train_loader):
        batched_fixed_patches = train_data['fixed_patches']
        batched_moving_patches = train_data['moving_patches']

        optimizer.zero_grad()

        predicted_theta = net(batched_moving_patches)

        predicted_deform = A.affine_transform(batched_moving_patches, predicted_theta)

        loss = criterion(batched_fixed_patches, predicted_deform, reduction='mean')

        loss.backward()
        patch_loss[batch_idx] = loss.item()
        print(loss.item())

        optimizer.step()

        if batch_idx % 1 == 0:
            print('====> Epoch: {}/{} \t Datapart: {}/{} \t Batch: {}/{}'  # \t Remaining time: Calculating ...'
                  .format(epoch + 1, epochs, set_idx + 1, num_sets - num_validation_sets, batch_idx + 1, int(fixed_patches.shape[0] / batch_size) + 1))

        cur_state_dict = net.state_dict()

    return patch_loss, cur_state_dict


def train_network(path_to_files, fix_set, mov_set, epochs, lr, batch_size, patch_size,
                  stride, fix_vols, mov_vols, loss_path, num_sets, device, model_name, voxelsize, num_validation_sets):

    net = Net().to(device)
    net.train()

    criterion = NCC().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    set_loss = torch.zeros(tot_num_sets - num_validation_sets).to(device)
    epoch_loss = torch.zeros(epochs).to(device)

    for epoch in range(epochs):

        # Print statements only valid for CUDA
        #print('Max memory allocated: ', (((torch.cuda.max_memory_allocated(device=device)/1024)/1024)/1024))
        #print('Max memory cached: ', (((torch.cuda.max_memory_cached(device=device)/1024)/1024)/1024))

        #print('Current memory allocated: ', (((torch.cuda.memory_allocated(device=device)/1024)/1024)/1024))
        #print('Current memory cached: ', (((torch.cuda.memory_cached(device=device)/1024)/1024)/1024))

        fix_set, mov_set, fix_vols, mov_vols = shuffle(fix_set, mov_set, fix_vols, mov_vols)

        # Defining validation set
        val_fix_set = fix_set[num_sets - num_validation_sets:num_sets]
        val_mov_set = mov_set[num_sets - num_validation_sets:num_sets]
        val_fix_vols = fix_vols[num_sets - num_validation_sets:num_sets]
        val_mov_vols = mov_vols[num_sets - num_validation_sets:num_sets]

        for set_idx in range(num_sets - num_validation_sets):

            print('Loading next set of volume data ...')
            vol_data = HDF5Image(path_to_files, fix_set[set_idx], mov_set[set_idx],
                                 fix_vols[set_idx], mov_vols[set_idx])
            vol_data.normalize()
            vol_data.to(device)

            print('Patching loaded data ...')
            patched_data, _ = create_patches(vol_data.data, patch_size, stride, device, voxelsize)

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
                                                       num_validation_sets
                                                       )

            set_loss[set_idx] = torch.mean(training_loss)

        model_info = {'patch_size': patch_size,
                      'state_dict': cur_state_dict}
        print('Saving model ... ')
        torch.save(model_info, model_name)

        print('Validating at the end of epoch {}'.format(epoch + 1))
        with torch.no_grad():
            validation_loss = validate(path_to_files, val_fix_set, val_mov_set, val_fix_vols, val_mov_vols,
                                       batch_size, net.eval(), criterion, device)

        epoch_loss = torch.mean(set_loss)

        with open(loss_path, mode='a') as loss:
            loss_writer = csv.writer(loss, delimiter=',')
            loss_writer.writerow([(epoch + 1), epoch_loss.item(), validation_loss.item()])

    return epoch_loss


if __name__ == '__main__':

    #=======================PARAMETERS==========================#
    lr = 1e-4  # learning rate
    epochs = 1  # number of epochs
    tot_num_sets = 4  # Total number of sets to use
    num_validation_sets = 1
    batch_size = 16
    patch_size = 30
    stride = 29
    voxelsize = 7.0000003e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #===========================================================#

    #=======================SAVING DATA=========================#
    now = datetime.now()
    date = now.strftime('%d%m%Y')
    time = now.strftime('%H%M%S')
    model_name = 'output/discard/model_{}_{}.pt'.format(date, time)
    loss_path = 'output/discard/loss_{}_epochs_{}_{}.csv'.format(epochs, date, time)

    path_to_files = '/users/kristofferroise/project/patient_data_proc/'

    # Filepath to dataset_information file
    path_to_info = '/users/kristofferroise/project/Diverse/'
    info_filename = 'dataset_information.csv'
    #===========================================================#

    #===================INITIALIZE FILES========================#
    with open(loss_path, 'w') as els:
        fieldnames = ['epoch', 'training_loss', 'validation_loss', 'lr=' + str(lr), 'batch_size=' + str(batch_size),
                      'patch_size=' + str(patch_size), 'stride=' + str(stride),
                      'number_of_datasets=' + str(tot_num_sets), 'device=' + str(device)]
        epoch_writer = csv.DictWriter(els, fieldnames=fieldnames)
        epoch_writer.writeheader()
    #===========================================================#

    dataset = GetDatasetInformation(path_to_info, info_filename)

    fix_set = dataset.fix_files
    mov_set = dataset.mov_files
    fix_vols = dataset.fix_vols
    mov_vols = dataset.mov_vols

    start_time = datetime.now()
    total_loss = train_network(path_to_files, fix_set, mov_set, epochs, lr, batch_size, patch_size,
                               stride, fix_vols, mov_vols, loss_path, tot_num_sets, device, model_name, voxelsize, num_validation_sets)
    print('Total time elapsed: ', datetime.now() - start_time)
