import torch
from torch.utils.data import Dataset, DataLoader
import csv
import math

# Folder dependent imports
from lib.network import Net
from lib.affine import affine_transform
from lib.HDF5Image import HDF5Image
from lib.patch_volume import create_patches


class CreatePredictionSet(Dataset):
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

def progress_printer(percentage):
    """Function returning a progress bar
        Args:
            percentage (float): percentage point
    """
    eq = '=====================>'
    dots = '......................'
    printer = '[{}{}]'.format(eq[len(eq) - math.ceil(percentage*20):len(eq)], dots[2:len(eq) - math.ceil(percentage*20)])
    return printer


def generate_patches(path_to_h5files, patch_size, stride, device, voxelsize):
    """Loading all datasets, creates patches and store all patches in a single array.
        Args:
            path_to_file (string): filepath to .txt file containing dataset information
            info_filename (string): filename for the above file
            path_to_h5files (string): path to .h5 files
            patch_size (int): desired patch size
            stride (int): desired stride between patches
            voxelsize (float): not used here, but create_patches has it as input
            tot_num_sets (int): desired number of sets to use in the model
        Returns:
            fixed patches: all fixed patches in the dataset ([num_patches, 1, **patch_size])
            moving patches: all moving patches in the dataset ([num_patches, 1, **patch_size])
    """

    fix_set = 'DataStOlavs19to28/p22_3115007/J65BP1R0_proc.h5'
    mov_set = 'DataStOlavs19to28/p22_3115007/J65BP1R2_proc.h5'
    fix_vols = '01'
    mov_vols = '12'

    print('Creating patches ... ')

    vol_data = HDF5Image(path_to_h5files, fix_set, mov_set,
                             fix_vols, mov_vols)
    vol_data.normalize()
    vol_data.cpu()

    patched_vol_data, loc = create_patches(vol_data.data, patch_size, stride, device, voxelsize)
    patched_vol_data = patched_vol_data.to(device)

    fixed_patches = patched_vol_data[:, 0, :].unsqueeze(1)
    moving_patches = patched_vol_data[:, 1, :].unsqueeze(1)

    return fixed_patches, moving_patches, loc


def create_net(model_name, device):
    net = Net().to(device)

    print('Loading weights ...')
    model = torch.load(model_name)
    net.load_state_dict(model['state_dict'])

    return net.eval()


def predict(path_to_h5files, patch_size, stride, device, voxelsize, model_name):

    path = 'output/txtfiles/prediction.csv'
    with open(path, 'w') as lctn:
        fieldnames = ['x_pos', 'y_pos', 'z_pos', 'theta']
        field_writer = csv.DictWriter(lctn, fieldnames=fieldnames)
        field_writer.writeheader()

    net = create_net(model_name, device)

    fixed_patches, moving_patches, loc = generate_patches(path_to_h5files, patch_size, stride, device, voxelsize)

    prediction_set = CreatePredictionSet(fixed_patches, moving_patches)
    prediction_loader = DataLoader(prediction_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    predicted_theta_full = torch.zeros([len(prediction_loader), 3, 4]).type(dtype)

    print('Predicting')

    for batch_idx, (fixed_batch, moving_batch) in enumerate(prediction_loader):

        fixed_batch, moving_batch = fixed_batch.to(device), moving_batch.to(device)

        predicted_theta = net(moving_batch)

        printer = progress_printer((batch_idx + 1) / len(prediction_loader))
        print(printer, end='\r')

        predicted_theta_full[batch_idx] = predicted_theta.type(dtype)
        printed_theta = predicted_theta_full[batch_idx].view(1, 12)

        with open(path, 'a') as lctn:
            lctn_writer = csv.writer(lctn, delimiter=',')
            lctn_writer.writerows((loc[batch_idx].cpu().numpy().round(5), printed_theta.numpy()))

    print('\n')
