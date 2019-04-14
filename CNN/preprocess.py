"""File with torch dataset instance and functions for preprocessing."""
import os

import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset



class FermiDataset(Dataset):

    def __init__(self, images_path, points_path, transform=None):
        self.images_path = images_path
        self.points_path = points_path
        self.index = sorted(os.listdir(self.images_path))
        self.transform = transform
        
    def __len__(self):
        return len(self.index)

    def create_target(self, y):
        mask = np.zeros((200, 200), dtype=np.int32)
        for coord in y:
            mask[coord[0], coord[1]]=1
        return np.array(mask, dtype=np.int64)


    def __getitem__(self, idx):
        
        image = np.load(os.path.join(self.images_path, self.index[idx]))
        image = image.reshape(*image.shape, -1)
        points = np.load(os.path.join(self.points_path, self.index[idx]))
        mask = np.array(self.create_target(points)).reshape(*image.shape[:2])
        sample = tuple([image, mask])
        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample[0], sample[1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h,
                    left: left + new_w]

        return tuple([image, mask])

class OneHotEncoding(object):
    def __init__(self, num_classes):
        if isinstance(num_classes, int):
            self.num_classes = num_classes
        else:
            raise ValueError('Num classes should be of the "int" type')

    def __call__(self, sample):
        image, mask = sample[0], sample[1]
        mask = np.eye(self.num_classes)[mask]
        return tuple([image, mask])

class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample[0], sample[1]
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return tuple([torch.from_numpy(image).to(torch.float32),
                      torch.from_numpy(mask).to(torch.float32)])