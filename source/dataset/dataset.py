from torch.utils.data import Dataset
import h5py
import os
import torch


class NucleiCellDataset(Dataset):

    def __init__(self, data_path, phase='train', transform=False,
                 image_size=512):
        """Custom PyTorch Dataset for hpa dataset.

        Parameters
        ----------
            data_path: str
                path to the nuclei dataset hdf5 file
            phase: str, optional
                phase this dataset is used for (train, val. test)
            transform:
                transform for the image data
            image_size: int
                the resolution of the image
        """
        if phase == "train":
            self.data_path = os.path.join(data_path,
                                          "train_" + str(image_size) + ".hdf5")
            self.target_nuclei_path = os.path.join(data_path, "train_" +
                                                   str(image_size) +
                                                   "_nuclei_mask.hdf5")
            self.target_cell_path = os.path.join(data_path, "train_" +
                                                 str(image_size) +
                                                 "_cell_mask.hdf5")
        elif phase == "validation":
            self.data_path = os.path.join(data_path, "validation_" +
                                          str(image_size) + ".hdf5")
            self.target_nuclei_path = os.path.join(data_path, "validation_"
                                                   + str(image_size) +
                                                   "_nuclei_mask.hdf5")
            self.target_cell_path = os.path.join(data_path, "validation_" +
                                                 str(image_size) +
                                                 "_cell_mask.hdf5")
        else:
            self.data_path = os.path.join(data_path, "test_" +
                                          str(image_size) + ".hdf5")
            self.target_nuclei_path = os.path.join(data_path, "test_" +
                                                   str(image_size) +
                                                   "_nuclei_mask.hdf5")
            self.target_cell_path = os.path.join(data_path, "test_" +
                                                 str(image_size) +
                                                 "_cell_mask.hdf5")
        self.phase = phase
        self.transform = transform

        self.target_dim = 2

        with h5py.File(self.data_path, "r") as h:
            self.data_names = list(h.keys())
            self.dim = 1

    def __len__(self):

        return len(self.data_names)

    def __getitem__(self, idx):

        with h5py.File(self.data_path, "r") as h:
            data = h[self.data_names[idx]][:]
        with h5py.File(self.target_nuclei_path, "r") as h:
            target_nuclei = h[self.data_names[idx]][:]
        with h5py.File(self.target_cell_path, "r") as h:
            target_cell = h[self.data_names[idx]][:]
        x = data.T

        y_nuclei = target_nuclei.T
        y_cell = target_cell.T

        x = torch.from_numpy(x).narrow(0, 0, 1)

        return x, y_nuclei, y_cell
