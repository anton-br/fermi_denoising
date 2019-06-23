"""File contains batch class named FermiBatch that allows to implement and run preprocessing. This preprocessing contains following actions:
1. load the data
2. generate mask from given points
3. crop image near with given points
4. normalize data (divide by max value on image)
5. prepare data for training.

Also, this file contains some functions for preprocessing results after training.
"""
import sys
sys.path.append('..')

import numpy as np

from batchflow import Dataset, ImagesBatch, FilesIndex, action, inbatch_parallel, Pipeline, any_action_failed, B, V
from batchflow.models.torch import TorchModel, UNet
from preprocessing import _crop_near_points

class FermiBatch(ImagesBatch):
    """Class for prepare Fermi images for training.
    
    Parameters
    ----------
    masks : array
        contains binary mask
    images : array
        Fermi images
    points : array
        coordinates of stars
    """
    def __init__(self, index, preloaded=None, *args, **kwargs):
        super().__init__(index, preloaded)
        self.masks = self.array_of_nones

    @property
    def array_of_nones(self):
        """ 1-D ndarray: ``NumPy`` array with ``None`` values."""
        return np.array([None] * len(self.index))
    
    components = 'images', 'points', 'masks'
    def _reraise_exceptions(self, results):
        """Check on errors."""
        if any_action_failed(results):
            all_errors = self.get_errors(results)
            raise RuntimeError("Cannot assemble the batch", all_errors)

    @action
    def load(self, fmt=None, components=None, *args, **kwargs):
        """Load data into components.
        
        Parameters
        ----------
        fmt : str
            Data format.
        components : str
            Component's name for loading data.
        """
        _ = args, kwargs
        if isinstance(components, str):
            components = list(components)
        if components is None:
            components = self.components[:-1]
        return self._load(fmt, components)

    @inbatch_parallel(init='indices', post="_assemble_load")
    def _load(self, ix, fmt, components):
        """Load data with index ```ix``` into components from ```components```.
        
        Parameters
        ----------
        ix : str
            Name of current image or mask.
        fmt : str
            Fomrat of data (here can be only 'fermi').
        components : list
            Component's name.
        
        Returns
        -------
            : list
            List with images and masks form file with index ```ix```.
        
        Note
        ----
        All actions with @inbatch_parallel decorator will be ran in parallel mode,
        it means that this function will be called
        """
        if isinstance(self.index, FilesIndex):
            path_x = self.index.get_fullpath(ix) 
            path_y = path_x.replace('/input_data/', '/reference_data/')
        else:
            raise ValueError("Source path is not specified")
        if fmt == "fermi":
            data = {}
            x_data = np.load(path_x)
            data['images'] = x_data.reshape(*x_data.shape, 1)
            if 'points' in components:
                data['points'] = np.load(path_y)
            return [data[comp] for comp in components]
        else:
            raise ValueError('Avalible values to `fmt` is `fermi` not {}'.fromat(fmt))

    def _assemble_load(self, results, *args, **kwargs):
        """Post function taht put data into components.
        
        Parameters
        ----------
        results : list
            list of results that comes from `inbatch_parallel` function.
        """
        _ = args, kwargs
        self._reraise_exceptions(results)
        components = kwargs.get("components", None)
        if components is None:
            components = self.components[:-1]
        for comp, data in zip(components, zip(*results)):
            data = np.array(data + (None, ))[:-1]
            setattr(self, comp, data)
        return self

    @action
    @inbatch_parallel(init='indices')
    def generate_masks(self, ix, mask_size=200, src=None, dst=None):
        """Generate mask with size ```mask_size``` from given ```ix```.
        
        Parameters
        ----------
        ix : int
            Position of data wich will be loaded load in batch.
        mask_size : int
            Size of created boolian mask.
        src : str
            component's name to data load
        dst : str
            component's name to data save
        """
        dst = dst if dst is not None else src

        i = self.get_pos(None, src, ix)
        points = getattr(self, src)[i]

        masks = np.zeros((mask_size, mask_size), dtype=int)
        for point in points:
            masks[point[0], point[1]] = 1.
        getattr(self, dst)[i] = masks
        return self
    
    @action
    @inbatch_parallel(init='indices')
    def random_crop_near_points(self, index, prob, output_size, src=None, dst=None):
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
        src : str
            component's name to data load
        dst : str
            component's name to data save
        """
        dst = dst if dst is not None else src

        ind = self.get_pos(None, src, index)
        image = getattr(self, src[0])[ind]
        mask = getattr(self, src[1])[ind]

        image, mask = _crop_near_points(image, mask, prob, output_size)

        getattr(self, dst[0])[ind] = image
        getattr(self, dst[1])[ind] = mask
        return self
    
    @action
    @inbatch_parallel(init='indices')
    def normalize(self, index, src=None, dst=None):
        """Normalize images devided my the most brightness pixel.
        
        Parameters
        ----------
        index : int
            Position of data wich will be loaded load in batch.
        src : str
            component's name to data load
        dst : str
            component's name to data save
        """
        dst = dst if dst is not None else src
        ind = self.get_pos(None, src, index)
        image = getattr(self, src)[ind]
        getattr(self, dst)[ind] = image / np.max(image)
        return self

    @action
    def prepare_tensors(self, src=None, dst=None, add_dim=False):
        """ Prepare data for training. Change the shape of data from (batch_size, )
        to (batch_szie,..., image_size, image_size) and add new dims if needed.
        
        Parameters
        ----------
        add_dim : bool
            if ture, new dim will be added to data.
            else nothing happend.
        src : str
            component's name to data load
        dst : str
            component's name to data save
        """
        dst = dst if dst is not None else src
        data = getattr(self, src)
        data = np.stack(data).astype(np.float32)
        if add_dim:
            data = np.expand_dims(data, axis=1)
            if len(data.shape) > 4:
                data = data.reshape(*data.shape[:4])
        if dst in self.components:
            setattr(self, dst, data)
        else:
            self.add_components(dst, init=data)
        return self
