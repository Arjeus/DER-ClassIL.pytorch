import pdb

import logging
import random
import cv2
import numpy as np
import os.path as osp
from copy import deepcopy
from PIL import Image
import multiprocessing as mp
from multiprocessing import Pool
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import pil_loader

from .dataset import get_dataset
from inclearn.tools.data_utils import construct_balanced_subset

logger = logging.getLogger(__name__)


def get_data_folder(data_folder, dataset_name):
    return osp.join(data_folder, dataset_name)


class IncrementalDataset:
    """Incremental generator of datasets.

    :param dataset_name: Name of the dataset, eg. "cifar100", "imagenet".
    :param random_order: Shuffle the class ordering, else use a deterministic
                         ordering.
    :param shuffle: Shuffle batch order between epochs.
    :param workers: Number of workers loading the data.
    :param batch_size: The batch size.
    :param seed: Random seed to use.
    :param increment: Number of class to add at each task.
    :param validation_split: Percent of training data to use for validation.
    :param onehot: Returns targets encoded as onehot vectors instead of scalars.
    :param initial_increment: Initial increment of classes, may be different
                              from the other increments.
    :param sampler: A sampler to use during training, for example
                    `torch.utils.data.DistributedSampler`. By default `None`.
    """

    def __init__(
        self,
        dataset_name,
        random_order=False,
        shuffle=True,
        workers=10,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.,
        onehot=False,
        initial_increment=None,
        sampler=None,
        data_folder="./data",
        trial_i=0,
    ):
        # The info about incremental split
        self.trial_i = trial_i
        self.start_class = 0
        #the number of classes for each step in incremental stage
        self.task_size = increment
        self.increments = []
        self.random_order = random_order
        self.validation_split = validation_split

        #-------------------------------------
        #Dataset Info
        #-------------------------------------
        self.data_folder = get_data_folder(data_folder, dataset_name)
        self.dataset_name = dataset_name
        self.train_dataset = None
        self.test_dataset = None
        self.n_tot_cls = -1
        
        # Get dataset class but pass the dataset_name string to _setup_data 
        dataset_class = get_dataset(dataset_name)
        self._setup_data(self.data_folder, self.dataset_name, self.random_order, self.trial_i)
        
        self._workers = workers
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._resampling = False
        
        #Currently, don't support multiple datasets
        self.train_transforms = dataset_class.train_transforms
        self.test_transforms = dataset_class.test_transforms
        #torchvision or albumentations
        self.transform_type = dataset_class.transform_type

        # memory Mt
        self.data_memory = None
        self.targets_memory = None
        # Incoming data D_t
        self.data_cur = None
        self.targets_cur = None
        # Available data \tilde{D}_t = D_t \cup M_t
        self.data_inc = None  # Cur task data + memory
        self.targets_inc = None
        # Available data stored in cpu memory.
        self.shared_data_inc = None
        self.shared_test_data = None

        #Current states for Incremental Learning Stage.
        self._current_task = 0

    def _setup_data(self, data_folder="./data", dataset_name="cifar100", random_order=False, trial_i=0):
        # FIXME: handles online loading of images
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []

        current_class_idx = 0  # When using multiple datasets
        train_dataset = get_dataset(dataset_name)(self.data_folder, train=True)
        test_dataset = get_dataset(dataset_name)(self.data_folder, train=False)
        self.train_dataset = train_dataset
        self.test_datasets = test_dataset
        self.n_tot_cls = self.train_dataset.n_cls  #number of classes in whole dataset

        self._setup_data_for_raw_data(train_dataset, test_dataset, current_class_idx)
        # !list
        self.data_train = np.concatenate(self.data_train)
        self.targets_train = np.concatenate(self.targets_train)
        self.data_val = np.concatenate(self.data_val)
        self.targets_val = np.concatenate(self.targets_val)
        self.data_test = np.concatenate(self.data_test)
        self.targets_test = np.concatenate(self.targets_test)

    def get_custom_loader_from_memory(self, class_indexes, mode="test"):
        if not isinstance(class_indexes, list):
            class_indexes = [class_indexes]
        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(self.data_memory,
                                                     self.targets_memory,
                                                     low_range=class_index,
                                                     high_range=class_index + 1)
            data.append(class_data)
            targets.append(class_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        return data, targets, self._get_loader(data, targets, shuffle=False, mode=mode)
    
    def _setup_data_for_raw_data(self, train_dataset, test_dataset, current_class_idx=0):
        increment = self.task_size

        x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
        x_val, y_val, x_train, y_train = self._split_per_class(x_train, y_train, self.validation_split)
        x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

        # Get Class Order
        order = [i for i in range(len(np.unique(y_train)))]
        if self.random_order:
            random.seed(self._seed)  # Ensure that following order is determined by seed:
            random.shuffle(order)
        elif get_dataset(self.dataset_name).class_order(self.trial_i) is not None:
            order = get_dataset(self.dataset_name).class_order(self.trial_i)

        self.class_order.append(order)

        y_train = self._map_new_class_index(y_train, order)
        y_val = self._map_new_class_index(y_val, order)
        y_test = self._map_new_class_index(y_test, order)

        y_train += current_class_idx
        y_val += current_class_idx
        y_test += current_class_idx

        current_class_idx += len(order)
        if self.start_class == 0:
            self.increments = [increment for _ in range(len(order) // increment)]
        else:
            self.increments.append(self.start_class)
            for _ in range((len(order) - self.start_class) // increment):
                self.increments.append(increment)
        self.data_train.append(x_train)
        self.targets_train.append(y_train)
        self.data_val.append(x_val)
        self.targets_val.append(y_val)
        self.data_test.append(x_test)
        self.targets_test.append(y_test)

    def _select_data_by_classes(self, class_indexes):
        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(self.data_train, self.targets_train, low_range=class_index, high_range=class_index + 1)
            data.append(class_data)
            targets.append(class_targets)
        return np.concatenate(data), np.concatenate(targets)

    def _setup_data_loaders(self):
        self.train_loader = self._get_loader(self.data_train, self.targets_train, mode="train")
        self.val_loader = self._get_loader(self.data_val, self.targets_val, shuffle=False, mode="test")
        self.test_loader = self._get_loader(self.data_test, self.targets_test, shuffle=False, mode="test")

    @property
    def n_tasks(self):
        return len(self.increments)

    def new_task(self, memory=None):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        min_class, max_class, x_train, y_train, x_test, y_test = self._get_cur_step_data_for_raw_data()

        self.data_cur, self.targets_cur = x_train, y_train

        if self.data_memory is not None:
            print("Set memory of size: {}.".format(len(self.data_memory)))
            if len(self.data_memory) != 0:
                x_train = np.concatenate((x_train, self.data_memory))
                y_train = np.concatenate((y_train, self.targets_memory))

        self.data_inc, self.targets_inc = x_train, y_train
        self.data_test_inc, self.targets_test_inc = x_test, y_test

        train_loader = self._get_loader(x_train, y_train, mode="train")
        val_loader = self._get_loader(x_test, y_test, shuffle=False, mode="test")
        test_loader = self._get_loader(x_test, y_test, shuffle=False, mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "increment": self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(x_train),
            "n_test_data": len(y_train),
        }

        self._current_task += 1
        return task_info, train_loader, val_loader, test_loader

    def get_custom_loader(
        self, class_indexes, memory=None, mode="test", data_source="train", sampler=None
    ):
        """Returns a custom loader.

        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if not isinstance(class_indexes, list):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_inc, self.targets_inc
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        elif data_source == 'specified' and memory is not None:
            x, y = memory
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets, = self._select(x, y, low_range=class_index, high_range=class_index + 1)
            data.append(class_data)
            targets.append(class_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        return data, targets, self._get_loader(data, targets, shuffle=False, mode=mode)

    def get_memory_loader(self, data_source="train"):
        if data_source == "train":
            x, y = self.data_memory, self.targets_memory
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        return self._get_loader(x, y, shuffle=False, mode="test")

    def get_few_samples_loader(self, n_samples, classes, mode="test", data_source="train"):
        data, targets = [], []
        for class_index in classes:
            class_data, class_targets = self._select(self.data_train, self.targets_train, low_range=class_index, high_range=class_index + 1)
            data.append(class_data[:n_samples])
            targets.append(class_targets[:n_samples])
        data = np.concatenate(data)
        targets = np.concatenate(targets)
        return self._get_loader(data, targets, shuffle=False, mode=mode)

    def get_empty_loader(self):
        return self._get_loader(np.array([]), np.array([]), shuffle=False, mode="test")

    def _get_cur_step_data_for_raw_data(self, ):
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])

        x_train, y_train = self._select(self.data_train, self.targets_train, low_range=min_class, high_range=max_class)
        x_test, y_test = self._select(self.data_test, self.targets_test, low_range=0, high_range=max_class)
        return min_class, max_class, x_train, y_train, x_test, y_test

    @staticmethod
    def _split_per_class(x, y, validation_split=0.0):
        """Splits train data for a subset of validation data.

        Split is done so that each class has a much data.
        """
        shuffled_indexes = np.random.permutation(x.shape[0])
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]

        x_val, y_val = [], []
        x_train, y_train = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])

        # !list
        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        return x_val, y_val, x_train, y_train

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))

    def _select(self, x, y, low_range=0, high_range=0):
        idxes = sorted(np.where(np.logical_and(y >= low_range, y < high_range))[0])
        if isinstance(x, list):
            selected_x = [x[idx] for idx in idxes]
        else:
            selected_x = x[idxes]
        return selected_x, y[idxes]

    def _get_loader(self, x, y, share_memory=None, shuffle=True, mode="train", batch_size=None, resample=None):
        if "balanced" in mode:
            x, y = construct_balanced_subset(x, y)

        batch_size = batch_size if batch_size is not None else self._batch_size

        if "train" in mode:
            trsf = self.train_transforms
            resample_ = self._resampling if resample is None else True
            if resample_ is False:
                sampler = None
            else:
                sampler = get_weighted_random_sampler(y)
            shuffle = False if resample_ is True else True
        elif "test" in mode:
            trsf = self.test_transforms
            sampler = None
        elif mode == "flip":
            if "imagenet" in self.dataset_name:
                trsf = A.Compose([A.HorizontalFlip(p=1.0), *self.test_transforms.transforms])
            else:
                trsf = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), *self.test_transforms.transforms])
            sampler = None
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))
        
        # Create memory flags - adding this to fix the ValueError issue
        if len(x) > 0:
            memory_flags = np.zeros(len(x), dtype=np.bool_)
        else:
            memory_flags = np.array([])
            
        return DataLoader(DummyDataset(x,
                                       y,
                                       memory_flags,
                                       trsf,
                                       trsf_type=self.transform_type,
                                       share_memory_=share_memory,
                                       dataset_name=self.dataset_name),
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=self._workers,
                          sampler=sampler,
                          pin_memory=True)


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, memory_flags, trsf, trsf_type='torchvision', share_memory_=None, dataset_name=None, open_image=False, data_type="train"):
        self.x, self.y = x, y
        self.memory_flags = memory_flags
        self.trsf = trsf
        self.trsf_type = trsf_type
        self.dataset_name = dataset_name
        self.open_image = open_image
        self.data_type = data_type
        self.share_memory = [None] * len(self) if len(self.x) > 0 else []

    def __len__(self):
        return self.x.shape[0] if hasattr(self.x, 'shape') else 0

    def __getitem__(self, idx):
        x, y, = self.x[idx], self.y[idx]
        # Handle iris dataset (tabular data)
        if self.dataset_name == 'iris':
            import torch
            x = torch.from_numpy(x).float()
            # reshape per‑sample from [N] → [N,1,1]
            x = x.unsqueeze(-1).unsqueeze(-1)
            return x, y, self.memory_flags[idx]
        memory_flag = self.memory_flags[idx]
        
        # Handle different dataset types
        if isinstance(x, np.ndarray) and len(x.shape) == 1:
            # This is for the digits dataset (flatten array)
            if self.trsf:
                # Convert to PIL image for transforms
                # Reshape to 8x8 (digits dataset shape)
                img = x.reshape(8, 8).astype(np.float32)
                # Scale to 0-255 range for PIL
                img = (img * 255 / 16).astype(np.uint8)  # digit images use 0-16 range
                x = Image.fromarray(img)
                x = self.trsf(x)
            else:
                # Just convert to tensor if no transform
                x = torch.from_numpy(x).float()
        elif isinstance(x, np.ndarray):
            # For image datasets like CIFAR
            x = Image.fromarray(x)
            if self.trsf:
                x = self.trsf(x)
        else:
            # For ImageNet
            if idx < len(self.share_memory):
                if self.share_memory[idx] is not None:
                    x = self.share_memory[idx]
                else:
                    x = cv2.imread(x)
                    x = x[:, :, ::-1]
                    self.share_memory[idx] = x
            else:
                x = cv2.imread(x)
                x = x[:, :, ::-1]
                
            if self.trsf_type and 'torch' in self.trsf_type:
                x = self.trsf(x)
            elif self.trsf:
                x = self.trsf(image=x)['image']
                
        return x, y, memory_flag


def _map_new_class_index(y, order):
    """Transforms targets for new class order."""
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_datasets(dataset_name):
    """Returns the dataset properly split. Also, if available, returns the
    properly shuffled class order.
    """
    return get_dataset(dataset_name)


def dataset_transforms(dataset_name, training=True, trsf_type='torch'):
    """Returns the dataset transforms.

    :param dataset_name: The name of the dataset.
    :param training: Whether to load the training or test data.
    :return: The transforms for the dataset.
    """
    dataset = get_dataset(dataset_name)

    if trsf_type == 'torch':
        if training:
            trsf = dataset.train_transforms
        else:
            trsf = dataset.test_transforms
    else:
        if training:
            trsf = dataset.train_al_transforms
        else:
            trsf = dataset.test_al_transforms

    return trsf
