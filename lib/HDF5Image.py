import h5py
import torch


class HDF5Image():
    """Class to load HDF5 images from .h5 files
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
        Returns:
            Return a variable data that contains both a fixed- and a moving image.
            The returned variable is on the form [2, x-length, y-length, z-length].
            volume_data.data[0, :] returns the fixed image.
            volume_data.data[1, :] returns the moving image.
        """

        fixed = '{}{}'.format(self.filepath, self.fix_file)
        moving = '{}{}'.format(self.filepath, self.mov_file)

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
        """ Transfers data-variable to specified device
        """
        self.data = self.data.to(device)

    def cpu(self):
        """ Transfers data-variable to CPU
        """
        self.data = self.data.cpu()
