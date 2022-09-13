from skimage import io, transform
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt

class SoNDataset(Dataset):
    """SoN scenicness dataset."""

    def __init__(self, im_paths,labels,attr_names, tr=None):
        self.im_paths = im_paths
        self.labels = labels
        self.label_avg = np.float32(labels.mean(axis=0))
        self.attr_names = attr_names
        self.transform = tr

    def __len__(self):
        return len(self.im_paths)


    def __getitem__(self, idx):
        img_name = self.im_paths[idx]
        try:
            image = io.imread(img_name)
        except:
            return None
        s = image.shape
        if len(s) > 3:
            print(img_name)
        if len(s) == 2:
            image = image[:, :, np.newaxis][:, :, [0, 0, 0]]
        if image.shape[2] != 3:
            image = image[:,:,0:3]
        image = np.ascontiguousarray(image)
        label = np.float32(self.labels[idx,:])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

class SUNAttributesDataset(Dataset):
    """SUN Attributes dataset."""

    def __init__(self, data_path,im_names,labels,attr_names, tr=None):
        self.data_path = data_path
        self.im_names = im_names
        self.labels = labels
        self.attr_names = attr_names
        self.transform = tr

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, 'images',
                                self.im_names[idx])
        image = io.imread(img_name)
        s = image.shape
        if len(s) > 3:
            print(img_name)
        if len(s) == 2:
            image = image[:, :, np.newaxis][:, :, [0, 0, 0]]
        if image.shape[2] != 3:
            image = image[:,:,0:3]
        image = np.ascontiguousarray(image)
        label = np.float32(self.labels[idx,:])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample