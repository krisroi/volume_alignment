import torch
from datetime import datetime

# Folder dependent imports
from lib.network import Net
import lib.affine as A
from lib.HDF5Image import HDF5Image
from lib.patch_volume import create_patches
from lib.ncc_loss import NCC
import lib.utils as ut


def create_net(model, device):
    net = Net().to(device)

    print('Loading weights ...')
    model = torch.load(model)
    net.load_state_dict(model['state_dict'])

    net.train()

    return net


def predict(warped_patch, model, device):

    net = create_net(model, device)

    predicted_theta = net(warped_patch)

    return predicted_theta


if __name__ == '__main__':

        # Defining filepath
    PROJ_ROOT = '/users/kristofferroise/'
    patient_group = 'project'
    patient = 'patient_data_proc'
    fixvol_no = 'vol01'
    movvol_no = 'vol01'

    mov_set = 'DataStOlavs9to18/p14_3191409/J44J72A0_proc.h5'
    fix_set = mov_set

    vol_data = HDF5Image(PROJ_ROOT, patient_group, patient, fix_set,
                         mov_set, fixvol_no, movvol_no)
    vol_data.normalize()

    # Creating known-transformation moving data
    theta_trans = torch.FloatTensor([[[0.98, 0, 0, -0.10],
                                      [0, 1, 0, 0.02],
                                      [0, 0, 1, -0.02]]])

    theta_idt = torch.FloatTensor([[[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]]])

    #=======================PARAMETERS==========================#
    patch_size = 30
    stride = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #===========================================================#

    print('Patching loaded data ...')
    patched_data = create_patches(vol_data.data, patch_size, stride)
    moving_patches = patched_data[:, 1, :].unsqueeze(1)

    warped_patch = torch.zeros(moving_patches.shape)

    for i in range(100, 117):
        warped_patch[i] = A.affine_transform(moving_patches[i, :].unsqueeze(0),
                                             theta_trans)
    #=======================SAVING DATA=========================#
    pred = True
    model = 'output/models/model_30102019_145550.pt'
    #===========================================================#
    if pred:
        start_time = datetime.now()
        with torch.no_grad():
            predicted_theta = predict(warped_patch[100:116, :], model, device)
            print(predicted_theta)
        print('Total time elapsed: ', datetime.now() - start_time)
