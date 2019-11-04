import h5py
import torch
import os
import numpy as np
import cv2


class HDF5Image():

    def __init__(self, PROJ_ROOT, patient_group, patient, fix_file, mov_file, fix_vol_no, mov_vol_no):
        super(HDF5Image, self).__init__()
        self.PROJ_ROOT = PROJ_ROOT
        self.patient_group = patient_group
        self.patient = patient
        self.fix_file = fix_file
        self.mov_file = mov_file
        self.fix_vol_no = 'vol{}'.format(fix_vol_no)
        self.mov_vol_no = 'vol{}'.format(mov_vol_no)

        self.data = self.load_hdf5()

    def load_hdf5(self):
        """ Loads HDF5-data from the specified filepath

        Returns:
            Return a variable data that contains both a fixed- and a moving image.
            The returned variable is on the form [2, x-length, y-length, z-length].
            volume_data.data[0, :] returns the fixed image.
            volume_data.data[1, :] returns the moving image.
        """

        fixed = os.path.join(self.PROJ_ROOT, '{}/{}/{}'.format(self.patient_group, self.patient, self.fix_file))
        moving = os.path.join(self.PROJ_ROOT, '{}/{}/{}'.format(self.patient_group, self.patient, self.mov_file))

        with h5py.File(fixed, 'r') as fix, h5py.File(moving, 'r') as mov:
            # Loads all volumes
            fixed_volumes = fix['CartesianVolumes']
            moving_volumes = mov['CartesianVolumes']

            # Sets vol01 equal to the data values in 'vol01'. Has shape (214, 214, 214) and type numpy.ndarray
            fix_vol = fixed_volumes[self.fix_vol_no][:]
            mov_vol = moving_volumes[self.mov_vol_no][:]

            shape = list(fix_vol.shape)
            shape = (2, shape[0], shape[1], shape[2])

            vol_data = torch.empty(shape)
            vol_data[0] = torch.from_numpy(fix_vol).float()
            vol_data[1] = torch.from_numpy(mov_vol).float()

        return vol_data

    def normalize(self):
        """ Normalizes pixel data in the .h5 files
        Example:
            volume_data = HDF5Image(required_parameters) # Loading files into volume_data
            volume_data.normalize() # Normalizes the values stored in volume_data
        """
        self.data = torch.div(self.data, torch.max(self.data))

    def to(self, device):
        self.data = self.data.to(device)


if __name__ == '__main__':
    PROJ_ROOT = '/users/kristofferroise/project'
    patient_group = 'patient_data/gr5_STolav5to8'
    patient = 'p7_3d'
    fixfile = 'J249J70K_proc.h5'
    movfile = 'J249J70M_proc.h5'
    fixvol_no = 'vol01'
    movvol_no = 'vol02'

    image = HDF5Image(PROJ_ROOT, patient_group, patient,
                      fixfile, movfile,
                      fixvol_no, movvol_no)

    image = np.array(image.data, dtype=np.uint8)

    cv2.imshow('X slice', image[1, int(image.shape[1] / 2), :, :])
    cv2.imshow('Y slice', image[1, :, int(image.shape[2] / 2), :])
    cv2.imshow('Z slice', image[1, :, :, int(image.shape[3] / 2)])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
