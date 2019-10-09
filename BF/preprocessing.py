import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from scipy.spatial import distance as sc_distance

import torch
from torch.utils.data import Dataset

def find_optimal_thr(model, images, masks, threshold, size, step):
    """Find optimal threshold from given threshold's.
    
    Parameters
    ----------
    model : torch model
        Trained model.
    images : nparray
        Array with images to predict.
    masks : nparray
        Array with answers to iamges.
    
    Returns
    -------
        : int
        optimal threshold
        : list
        distance with each threshold
    """
    g_dist = []
    for tr in threshold:
        dist = []
        for img, msk in zip(images, masks):
            img_tens = torch.Tensor(list_crop(img.reshape(1, 200, 200), size, step))
            img_sigm = sigmoid(model.model(img_tens.to('cuda')).cpu()
                               .detach().numpy()).transpose(0, 2, 3, 1)[:,:,:,1]
            assemble_img = assemble_imgs(img_sigm, (200, 200), step)
            filt_pred = filter_prediction(np.array(assemble_img > tr))
            img = np.array(np.where(filt_pred > tr)).T
            mask = np.array(np.where(msk)).T
            dist.append(calculate_distance(img, mask))
        g_dist.append(np.mean(dist))
    return g_dist, threshold[np.argmin(g_dist)]

def list_crop(image, size, step):
    """Crop image on many images with size equal to ```size```.
    
    Parameters
    ----------
    image : 2D ndarray
        Image to crop.
    size : int
        Size of croped image.
    
    Returns
    -------
        : list
        List with images size ```size```.
        """
    imgs = []
    for i in range(0, 200-size+1, step):
        for j in range(0, 200-size+1, step):
            imgs.append(image[:, i : i+size, j : j+size].reshape(-1, size, size))
    return imgs

def assemble_imgs(list_imgs, out_size, step):
    """Assemble croped images.
    
    list_imgs : list
        Croped images to assemble.
    out_size : int
        Size of assembled image.
    
    Re sturns
    -------
        : 2d ndarray
        Assembled image.
    """
    image = np.zeros((out_size))
    norm = np.zeros((out_size))
    size = list_imgs[0].shape[1]
    num_img = 0
    for i in range(0, out_size[0]-size+1, step):
        for j in range(0, out_size[1]-size+1, step):
            image[i : i+size, j : j+size] += list_imgs[num_img]#.reshape(list_imgs[0].shape[1:])
            norm[i : i+size, j : j+size] += 1
            num_img += 1
    return image/norm

def filter_prediction(predict):
    """Searches for points that are closer than 2 pixels to each other
    and replaces them with 1 pixel.
    
    Parameters
    ----------
    predict : ndarray of 2d ndarrays
        Array with predicted masks.
    
    Returns
        : ndarray of 2d ndarrays
        Filtered predictions.
    """
    points = np.array(np.where(predict)).T
    new_points = []
    start_ix = -1
    if len(points) == 1:
        return predict
    for ix_p, _ in enumerate(points):
        same_val = [points[ix_p]]
        if ix_p < start_ix:
            continue
        for ix_np in range(ix_p+1, len(points)):
            if np.sum(np.abs(points[ix_p] - points[ix_np])) < 4:
                same_val.append(points[ix_np])
                if ix_np == len(points)-1:
                    ix_np += 1
            else:
                break
        new_points.append(list(np.round(np.mean(same_val, axis=0)).astype(int)))
        start_ix = ix_np
    return create_mask(np.array(new_points))

def sigmoid(x):
    """Sigmoid function.
    
    Parameters
    ----------
    x : int, float or array
        input data.
        
    Returns
    -------
        : the same as input type
        \frac{1}{1 + \exp_{-x}}.
    """
    return 1 / (1 + np.exp(-x))

def calculate_distance(true_coords_vector, coords_vector):
    """Distance between given vectors.
    
    Parameters
    ----------
    true_coords_vector : ndarray of 2d ndarrays
        Array with source coordinates.
    coords_vector : ndarray of 2d ndarrays
        Array with predicted coordinates.
    
    Returns
    -------
        : float
        Distance between vectors.
    """
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

def create_mask(y):
    """Create mask from list of points.
    
    Parameters
    ----------
    y : ndarray
        Coordinates of source.
    
    Returns
    -------
        : 2d ndarray
        mask with sources.
    """
    mask = np.zeros((200, 200), dtype=np.int32)
    for coord in y:
        mask[coord[0], coord[1]] = 1
    return np.array(mask, dtype=np.int64)

def draw_plots(lenght, dset, titles, hspace=-0.1, figsize=(10, 10)):
    """Draw examples of data. 
    
    Parameters
    ----------
    lenght : int
        Number of items to draw.
    dset : ndarray
        Data to draw.
    titles : array with lenght 2
        Title for each item.
    hspace : float
        Parameter for plt.subplots_adjust
    figsize : array with lenght 2
        Size of plots.
    """
    num_subplots = len(titles)
    _, ax = plt.subplots(num_subplots, lenght, figsize=figsize)
    indices = np.random.choice(range(len(dset)), replace=False, size=lenght)
    for ix, i in enumerate(indices):
        sample = dset[i]
        reshape_size = np.array([np.max(sample[0].shape)] * 2)
        plt.tight_layout()
        for j, (title, spl) in enumerate(zip(titles, sample)):
            if reshape_size.prod() != np.array(spl.shape).prod():
                spl = spl.reshape(*reshape_size, -1)
                spl = spl[:, :, 1]
            ax[j][ix].grid()
            ax[j][ix].set_title(title + ' #{}'.format(i))
            np,
            ax[j][ix].imshow(spl.reshape(*reshape_size), cmap='gray')
        plt.subplots_adjust(wspace=0.4, hspace=hspace)
    plt.show()

    
class FermiDataset(Dataset):
    """Class with functions to 
    sample fermi data.
    
    Parameters
    ----------
    images_path : str
        Path to images.
    points_path : str
        Path to sources.
    """
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
        image = image
        points = np.load(os.path.join(self.points_path, self.index[idx]))
        mask = np.array(create_mask(points)).reshape(*image.shape[:2])
        sample = tuple([image, mask])
        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomCropNearPoints(object):
    """Crop near points.
    
    Parameters
    ----------
    output_size : array or int
        Size of cropped image.
    prob : float
        Probability of sample image with soruce.
    """
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
        image, mask = _crop_near_points(image, mask, self.prob, self.output_size)
        return tuple([image, mask])

def _crop_near_points(image, mask, prob, output_size):
    """Crop given image on size ```output_size``` and with probability ```prob```
    this function crop image with no less then one source on it.

    Parameters
    ----------
    index : int
        Position of data wich will be loaded load in batch.
    prob : float
        Probability of sample image with soruce.
    output_size : int or list or tuple with lenght 2.
        Size of resulted image.
    """
    shape = image.shape[:2]
    output_size = output_size if isinstance(output_size, (list, tuple)) else (output_size, output_size)
    points = np.array(np.where(mask)).T
    if np.random.random() > prob or len(points) == 0:
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
    return image, mask
