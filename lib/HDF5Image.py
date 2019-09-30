import h5py
import torch
import os


def load_hdf5(fixed_file, moving_file):
    PROJ_ROOT = '/users/kristofferroise/project'
    patient_group = 'patient_data/gr5_STolav5to8'
    patient = 'p7_3d'

    fixed = os.path.join(PROJ_ROOT, '{}/{}/{}'.format(patient_group, patient, fixed_file))
    moving = os.path.join(PROJ_ROOT, '{}/{}/{}'.format(patient_group, patient, moving_file))

    num_volumes = 2

    with h5py.File(fixed, 'r') as fix, h5py.File(moving, 'r') as mov:
        # Loads all volumes
        fixed_volumes = fix['CartesianVolumes']
        moving_volumes = mov['CartesianVolumes']

        # Sets vol01 equal to the data values in 'vol01'. Has shape (214, 214, 214) and type numpy.ndarray
        fix_vol01 = fixed_volumes['vol01'][:]
        mov_vol01 = moving_volumes['vol02'][:]

        shape = list(mov_vol01.shape)
        shape = (num_volumes, shape[0], shape[1], shape[2])

        data = torch.empty(shape)
        print(data.shape)

        data[0] = torch.from_numpy(fix_vol01)
        data[1] = torch.from_numpy(mov_vol01)

    return data, shape
