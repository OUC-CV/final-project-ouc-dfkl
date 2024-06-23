import h5py
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr_image = np.expand_dims(f['lr'][idx][:51, :51] / 255., axis=0)
            hr_image = np.expand_dims(f['hr'][idx][:51, :51] / 255., axis=0)
            return lr_image, hr_image

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr_image = np.expand_dims(f['lr'][str(idx)][:51, :51] / 255., axis=0)
            hr_image = np.expand_dims(f['hr'][str(idx)][:51, :51] / 255., axis=0)
            return lr_image, hr_image

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
