"""File with torch dataset instance and functions for preprocessing."""
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from scipy.spatial import distance as sc_distance

import torch
from torch.utils.data import Dataset


def calculate_distance(true_coords_vector, coords_vector):
    """calculate metric"""
    if len(coords_vector) == 0 and len(true_coords_vector) != 0:
        return np.sum(sc_distance.cdist(np.array([[100, 100]]), true_coords_vector,
                                        'euclidean')) + 16 * len(true_coords_vector)
    if len(coords_vector) != 0 and len(true_coords_vector) == 0:
        return np.sum(sc_distance.cdist(np.array([[100, 100]]), coords_vector,
                                        'euclidean')) + 16 * len(coords_vector)
    if len(coords_vector) == 0 and len(true_coords_vector) == 0:
        return 0
    true_coords = np.vstack(tuple({tuple(row) for row in true_coords_vector}))
    coords = np.vstack(tuple({tuple(row) for row in coords_vector}))
    tree_true_coords = BallTree(true_coords)
    tree_coords = BallTree(coords)
    distance_from_true_array, _ = tree_true_coords.query(coords)
    distance_from_found_array, _ = tree_coords.query(true_coords)
    distance_from_found = np.sum(distance_from_found_array)
    distance_from_true = np.sum(distance_from_true_array)
    return np.sum(distance_from_found_array) + np.sum(distance_from_true_array)

def create_target(y):
    """create mask from list of points"""
    mask = np.zeros((200, 200), dtype=np.int32)
    for coord in y:
        mask[coord[0], coord[1]] = 1
    return np.array(mask, dtype=np.int64)

def draw_plots(lenght, dset, titles):
    """draw examples of data"""
    num_subplots = len(titles)
    _, ax = plt.subplots(num_subplots, lenght, figsize=(15, 15))
    indices = np.random.choice(range(len(dset)), replace=False, size=lenght)
    for ix, i in enumerate(indices):
        sample = dset[i]
        reshape_size = sample[0].shape[:2]
        plt.tight_layout()
        for j, (title, spl) in enumerate(zip(titles, sample)):
            ax[j][ix].grid()
            ax[j][ix].set_title(title + ' #{}'.format(i))
            ax[j][ix].imshow(spl.reshape(*reshape_size))
        plt.subplots_adjust(wspace=0.1, hspace=-0.6)

def plot_stats(loss, acc, name):
    """draw train and test statistics"""
    _, ax = plt.subplots(1, 2, figsize=(20, 7))
    ax[0].plot(loss)
    ax[0].set_title(name + ' Loss')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Loss')
    ax[1].plot(acc)
    ax[1].set_title(name + ' Distance')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Distance')
    plt.show()

class FermiDataset(Dataset):
    """sample fermi data"""
    def __init__(self, images_path, points_path, transform=None):
        self.images_path = images_path
        self.points_path = points_path
        self.index = sorted(os.listdir(self.images_path))
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        image = np.load(os.path.join(self.images_path, self.index[idx]))
        image = image.reshape(*image.shape, -1)
        points = np.load(os.path.join(self.points_path, self.index[idx]))
        mask = np.array(create_target(points)).reshape(*image.shape[:2])
        sample = tuple([image, mask])
        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomCrop(object):
    """random crop"""
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

class RandomCropNearPoints(object):
    """crop near points"""
    def __init__(self, output_size, prob):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample[0], sample[1]

        shape = image.shape[:2]
        output_size = self.output_size

        points = np.array(np.where(sample[1])).T

        if np.random.random() > self.prob or len(points) == 0:
            #usiual crop
            top = np.random.randint(0, shape[0] - output_size[0])
            left = np.random.randint(0, shape[1] - output_size[1])
            top = [top, top + output_size[0]]
            left = [left, left + output_size[1]]
        else:
            #crop near random point
            ix = np.random.choice(np.arange(len(points)))
            point = points[ix]
            coords = []
            for i in range(2):
                if point[i] == 0:
                    coords.append([0, output_size[i]])
                elif point[i] == 199:
                    coords.append([shape[i] - output_size[i], shape[i]])
                elif point[i] > shape[i] / 2:
                    if point[i] + output_size[i] < shape[i]:
                        w = point[i] + np.random.randint(1, output_size[i])
                    else:
                        w = point[i] + np.random.randint(1, shape[i] - point[i])
                    coords.append([w - output_size[i], w])
                else:
                    if point[i] - output_size[i] > 0:
                        w = point[i] - np.random.randint(0, output_size[i])
                    else:
                        w = np.random.randint(0, point[i])
                    coords.append([w, w + output_size[i]])
            top, left = coords[0], coords[1]

        image = image[top[0]: top[1],
                      left[0]: left[1]]

        mask = mask[top[0]: top[1],
                    left[0]: left[1]]


        return tuple([image, mask])

class OneHotEncoding(object):
    """ohe"""
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
    """create torch tensors"""
    def __call__(self, sample):
        image, mask = sample[0], sample[1]
        image = image.transpose((2, 0, 1))
        return tuple([torch.from_numpy(image).to(torch.float32),
                      torch.from_numpy(mask).to(torch.float32)])
