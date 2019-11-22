import torch
from torch.utils.data import Dataset, DataLoader
import csv
import math
import sys
import os
from datetime import datetime

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

    def __init__(self, fixed_patches, moving_patches, patch_location):
        self.fixed_patches = fixed_patches
        self.moving_patches = moving_patches
        self.patch_location = patch_location

    def __len__(self):
        return self.fixed_patches.shape[0]

    def __getitem__(self, idx):
        return self.fixed_patches[idx, :], self.moving_patches[idx, :], self.patch_location[idx, :]

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

    fixed_patches = patched_vol_data[:, 0, :].unsqueeze(1)
    moving_patches = patched_vol_data[:, 1, :].unsqueeze(1)

    return fixed_patches, moving_patches, loc


def create_net(model_name, device):
    """Load network model and return in .eval() mode
        Args:
            model_name (string): absolute path to model
            device (torch.device): device to load model on
    """
    net = Net().to(device)

    print('Loading weights ...')
    model = torch.load(model_name)
    net.load_state_dict(model['model_state_dict'])

    return net.eval()


def predict(path_to_h5files, patch_size, stride, device, voxelsize, model_name, batch_size):

    """Predict global transformation on a prediction set
        Args:
            path_to_h5files (string): absolute path to folder holding .h5 files
            patch_size (int)
            stride (int)
            device (torch.device): device to run prediction on
            voxelsize (float): size of each voxel in a patch
            model_name (string): absolute path to model
            batch_size(int)
    """
    net = create_net(model_name, device)

    model_name = os.path.splitext(model_name)[0] # Remove extension to save only model name with prediction

    loc_path = 'output/txtfiles/{}_loc_prediction.csv'.format(model_name.rsplit('/', 1)[-1]) # Path to location prediction with model name
    theta_path = 'output/txtfiles/{}_theta_prediction.csv'.format(model_name.rsplit('/', 1)[-1]) # Path to theta prediction with model name

    with open(loc_path, 'w') as lctn:
        fieldnames = ['x_pos', 'y_pos', 'z_pos']
        field_writer = csv.DictWriter(lctn, fieldnames=fieldnames)
        field_writer.writeheader()

    with open(theta_path, 'w') as tht:
        fieldnames = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
        field_writer = csv.DictWriter(tht, fieldnames=fieldnames)
        field_writer.writeheader()

    fixed_patches, moving_patches, loc = generate_patches(path_to_h5files, patch_size, stride, device, voxelsize)

    print('\n')
    print('Number of prediction samples: {}'.format(fixed_patches.shape[0]))
    print('\n')

    prediction_set = CreatePredictionSet(fixed_patches, moving_patches, loc)
    prediction_loader = DataLoader(prediction_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print('Predicting')

    for batch_idx, (fixed_batch, _, loc) in enumerate(prediction_loader):

        printer = progress_printer((batch_idx + 1) / len(prediction_loader))
        print(printer, end='\r')

        predicted_theta_tmp = torch.zeros([len(prediction_loader), fixed_batch.shape[0], 12]).type(dtype)
        loc_tmp = torch.zeros([len(prediction_loader), fixed_batch.shape[0], 3]).type(dtype)

        moving_batch = moving_batch.to(device)

        predicted_theta = net(moving_batch)
        predicted_theta = predicted_theta.view(-1, 12)

        predicted_theta_tmp[batch_idx] = predicted_theta.type(dtype)
        loc_tmp[batch_idx] = loc.type(dtype)

    with open(loc_path, 'a') as lctn:
        lctn_writer = csv.writer(lctn, delimiter=',')
        lctn_writer.writerows((loc_tmp.cpu().numpy().round(5)))

    with open(theta_path, 'a') as tht:
        theta_writer = csv.writer(tht)
        theta_writer.writerows((predicted_theta_tmp.cpu().numpy()))


    print('\n')


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

    model_name = str(sys.argv[1]) # Run predict with modelname from training as argument
    path_to_h5files = '/mnt/EncryptedFastData/krisroi/patient_data_proc/'
    patch_size = 50
    stride = 70
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    voxelsize = 7.0000003e-4
    batch_size = 32

    with torch.no_grad():
        prediction_start_time = datetime.now()
        predict(path_to_h5files, patch_size, stride, device, voxelsize, model_name, batch_size)
        print('Prediction runtime: ', datetime.now() - prediction_start_time)
        print('\n')
