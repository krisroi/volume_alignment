import h5py
import torch
import numpy as np


class HDF5Image():
    """Loading .h5 files and returns them in a 2-channel tensor, one fixed- and one moving channel.
        Args:
            filepath (string): absolute path to .h5 files
            fix_file (string): relative path to specific fixed file
            mov_file (string): relative path to specific moving file
            fix_vol_no (string): specific volume to extract from the fixed file
            mov_vol_no (string): specific volume to extract from the moving file
        Returns:
            A tensor that contains both a fixed- and a moving image.
            The returned tensor is on the form [2, x-length, y-length, z-length].
            volume_data.data[0, :] returns the fixed image.
            volume_data.data[1, :] returns the moving image.
    """

    def __init__(self, filepath, fix_file, mov_file, fix_vol_no, mov_vol_no):
        super(HDF5Image, self).__init__()
        self.filepath = filepath
        self.fix_file = fix_file
        self.mov_file = mov_file
        self.fix_vol_no = 'vol{}'.format(fix_vol_no)
        self.mov_vol_no = 'vol{}'.format(mov_vol_no)

        self.data = self.load_hdf5()

    def load_hdf5(self):
        """ Loads HDF5-data from the specified filepath
        """

        fixed = '{}{}'.format(self.filepath, self.fix_file)
        moving = '{}{}'.format(self.filepath, self.mov_file)

        with h5py.File(fixed, 'r') as fix, h5py.File(moving, 'r') as mov:

            fixed_volumes = fix['CartesianVolumes']
            moving_volumes = mov['CartesianVolumes']

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

    def histogram_equalization(self):
        # Preforms a histogram_equalization to the data.
        data = self.data.numpy().astype('uint8')
        for i in range(data.shape[0]):
            hist, bins = np.histogram(data[i].flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')
            data[i] = cdf[data[i]]
        self.data = torch.from_numpy(data).float()

    def to(self, device):
        """ Transfers data-variable to specified device
            Args:
                device (torch.device): desired device
        """
        self.data = self.data.to(device)

    def cpu(self):
        """ Transfers data-variable to CPU
        """
        self.data = self.data.cpu()
