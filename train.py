import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import math
import csv
from datetime import datetime
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Folder dependent imports
from lib.network import Net
import lib.affine as A
from lib.HDF5Image import HDF5Image
from lib.patch_volume import create_patches
from lib.ncc_loss import NCC
import lib.utils as ut
from lib.data_info_loader import GetDatasetInformation


class CreateDataset(Dataset):
    """Reads fixed- and moving patches and returns them as a Dataset object for
        use with Pytorch's handy DataLoader.
        Args:
            fixed_patches (Tensor): Tensor containing the fixed patches
            moving_patches (Tensor): Tensor containing the moving patches
        Example:
            dataset = CreateDataset(fixed_patches, moving_patches)
            dataloader = DataLoader(dataset, **kwargs)
    """

    def __init__(self, fixed_patches, moving_patches):
        self.fixed_patches = fixed_patches
        self.moving_patches = moving_patches

    def __len__(self):
        return self.fixed_patches.shape[0]

    def __getitem__(self, idx):
        return self.fixed_patches[idx, :], self.moving_patches[idx, :]


def validate(path_to_files, val_fix_set, val_mov_set, val_fix_vols, val_mov_vols, batch_size, net, criterion, device):

    avg_val_loss = torch.zeros(num_validation_sets, device=device)  # Holding average validation loss over all validation sets
    step_val_loss = torch.Tensor([])  # Holding validation loss over each batch_idx for all validation sets

    for val_set_idx in range(len(val_fix_set)):

        # Loading and normalizing validation images from .h5 files
        validation_data = HDF5Image(path_to_files, val_fix_set[val_set_idx], val_mov_set[val_set_idx],
                                    val_fix_vols[val_set_idx], val_mov_vols[val_set_idx])
        validation_data.normalize()
        validation_data.cpu()

        patched_validation_data, _ = create_patches(validation_data.data, patch_size, stride, device, voxelsize)

        fixed_patches = patched_validation_data[:, 0, :].unsqueeze(1)
        moving_patches = patched_validation_data[:, 1, :].unsqueeze(1)

        # Create and load validation data in batches of batch_size
        validation_set = CreateDataset(fixed_patches, moving_patches)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        set_val_loss = torch.zeros(len(validation_loader), device=device)  # Holding validation loss over all batch_idx for one validation set

        for batch_idx, (fixed_batch, moving_batch) in enumerate(validation_loader):

            fixed_batch, moving_batch = fixed_batch.to(device), moving_batch.to(device)  # Transfer patches to GPU (if available)

            predicted_theta = net(moving_batch)  # Forward pass to predict deformation matrix

            predicted_deform = A.affine_transform(moving_batch, predicted_theta)  # Affine transformation to predict moving image

            loss = criterion(fixed_batch, predicted_deform, reduction='mean')  # Compute NCC loss between fixed_patches and predicted moving_patches

            set_val_loss[batch_idx] = loss.item()  # Store loss

        avg_val_loss[val_set_idx] = torch.mean(set_val_loss)
        step_val_loss = torch.cat((step_val_loss, set_val_loss), dim=0)

    return avg_val_loss, step_val_loss


def train(patched_data, epoch, epochs, lr, batch_size, net, criterion,
          optimizer, set_idx, num_sets, device, num_validation_sets):

    fixed_patches = patched_data[:, 0, :, ].unsqueeze(1)
    moving_patches = patched_data[:, 1, :].unsqueeze(1)

    train_set = CreateDataset(fixed_patches, moving_patches)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    set_train_loss = torch.zeros(len(train_loader), device=device)  # Holding training loss over all batch_idx for one training set

    for batch_idx, (fixed_batch, moving_batch) in enumerate(train_loader):

        fixed_batch, moving_batch = fixed_batch.to(device), moving_batch.to(device)

        optimizer.zero_grad()

        predicted_theta = net(moving_batch)

        predicted_deform = A.affine_transform(moving_batch, predicted_theta)

        loss = criterion(fixed_batch, predicted_deform, reduction='mean')
        loss.backward()
        optimizer.step()

        set_train_loss[batch_idx] = loss.item()

        if batch_idx % 10 == 0:
            print('{}/{}%'.format(np.round((batch_idx / len(train_loader)) * 100, 3), 100), end='\r')

        cur_state_dict = net.state_dict()

    return set_train_loss, cur_state_dict


def train_network(path_to_files, fix_set, mov_set, epochs, lr, batch_size, patch_size,
                  stride, fix_vols, mov_vols, avg_loss_path, num_sets, device, model_name, voxelsize, num_validation_sets):

    net = Net().to(device)

    criterion = NCC().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)  # To change lr during training

    # Defining training set
    train_fix_set = fix_set[0:num_sets - num_validation_sets]
    train_mov_set = mov_set[0:num_sets - num_validation_sets]
    train_fix_vols = fix_vols[0:num_sets - num_validation_sets]
    train_mov_vols = mov_vols[0:num_sets - num_validation_sets]

    # Defining validation set
    val_fix_set = fix_set[num_sets - num_validation_sets:num_sets]
    val_mov_set = mov_set[num_sets - num_validation_sets:num_sets]
    val_fix_vols = fix_vols[num_sets - num_validation_sets:num_sets]
    val_mov_vols = mov_vols[num_sets - num_validation_sets:num_sets]

    # Creating loss-storage variables
    avg_training_loss = torch.zeros(num_sets - num_validation_sets).to(device)
    step_training_loss = torch.Tensor([]).to(device)
    total_step_loss = torch.Tensor([]).to(device)
    epoch_loss = torch.zeros(epochs).to(device)

    print('Initializing...', flush=True)

    for epoch in range(epochs):

        train_fix_set, train_mov_set, train_fix_vols, train_mov_vols = shuffle(train_fix_set, train_mov_set, train_fix_vols, train_mov_vols)

        for set_idx in range(num_sets - num_validation_sets):

            train_data = HDF5Image(path_to_files, train_fix_set[set_idx], train_mov_set[set_idx],
                                   train_fix_vols[set_idx], train_mov_vols[set_idx])
            train_data.normalize()
            train_data.cpu()

            patched_training_data, _ = create_patches(train_data.data, patch_size, stride, device, voxelsize)

            training_loss, cur_state_dict = train(patched_training_data,
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

            avg_training_loss[set_idx] = torch.mean(training_loss)
            step_training_loss = torch.cat((step_training_loss, training_loss), dim=0)

            print('Intermediate loss per training set: {} \t Steps in this set: {}'.format(avg_training_loss[set_idx], math.ceil(patched_training_data.shape[0] / batch_size)))

        model_info = {'patch_size': patch_size,
                      'state_dict': cur_state_dict}
        torch.save(model_info, model_name)

        with torch.no_grad():
            avg_validation_loss, step_validation_loss = validate(path_to_files, val_fix_set, val_mov_set, val_fix_vols, val_mov_vols,
                                                                 batch_size, net.eval(), criterion, device)

        avg_epoch_loss = torch.mean(avg_training_loss)
        print('\n Epoch: {}/{} \t Training_loss: {} \t Validation loss: {}'
              .format(epoch + 1, epochs, avg_epoch_loss, avg_validation_loss.item()))

        with open(avg_loss_path, mode='a') as loss:
            loss_writer = csv.writer(loss, delimiter=',')
            loss_writer.writerow([(epoch + 1), avg_epoch_loss.item(), torch.mean(avg_validation_loss).item()])

        total_step_loss = torch.cat((total_step_loss, step_training_loss))

    return total_step_loss


if __name__ == '__main__':

    #=======================PARAMETERS==========================#
    lr = 1e-3  # learning rate
    epochs = 1  # number of epochs
    tot_num_sets = 2  # Total number of sets to use (25 max)
    num_validation_sets = 1
    batch_size = 1
    patch_size = 40
    stride = 35
    voxelsize = 7.0000003e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #===========================================================#

    #=======================SAVING DATA=========================#
    now = datetime.now()
    date = now.strftime('%d%m%Y')
    time = now.strftime('%H%M%S')
    model_name = 'output/discard/model_{}_{}.pt'.format(date, time)
    avg_loss_path = 'output/discard/avg_loss_{}_epochs_{}_{}.csv'.format(epochs, date, time)
    step_loss_path = 'output/discard/step_loss_{}_epochs_{}_{}.csv'.format(epochs, date, time)

    path_to_files = '/users/kristofferroise/project/patient_data_proc/'

    # Filepath to dataset_information file
    path_to_info = '/users/kristofferroise/project/Diverse/'
    info_filename = 'dataset_information.csv'
    #===========================================================#

    #===================INITIALIZE FILES========================#
    with open(avg_loss_path, 'w') as els:
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

    fix_set, mov_set, fix_vol, mov_vols = shuffle(fix_set, mov_set, fix_vols, mov_vols)

    fix_set = fix_set[0:tot_num_sets]
    mov_set = mov_set[0:tot_num_sets]
    fix_vols = fix_vols[0:tot_num_sets]
    mov_vols = mov_vols[0:tot_num_sets]

    start_time = datetime.now()
    total_step_loss = train_network(path_to_files, fix_set, mov_set, epochs, lr, batch_size, patch_size,
                                    stride, fix_vols, mov_vols, avg_loss_path, tot_num_sets, device, model_name, voxelsize, num_validation_sets)
    print('Total time elapsed: ', datetime.now() - start_time)
