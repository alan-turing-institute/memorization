# -*- coding: utf-8 -*-

"""
Wrapper around a dataset that facilitates creating cross-validation splits.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import os
import sys
import torch

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

from torch.utils.data import Subset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from constants import LOGIT_LAMBDA
from seed_generator import SeedGenerator

KNOWN_DATASETS = [
    "CIFAR10",
    "CelebA",
    "BinarizedMNIST",
]


def dataset_with_indices(klas):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        data, target = klas.__getitem__(self, index)
        return data, target, index

    t = type(klas.__name__, (klas,), {"__getitem__": __getitem__})
    return t


class Binarize:
    # Based on:
    # https://github.com/NVlabs/NVAE/blob/38eb9977aa6859c6ee037af370071f104c592695/datasets.py
    def __call__(self, tensor):
        return torch.Tensor(tensor.size()).bernoulli_(tensor)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class UniformDequantization:
    def __call__(self, tensor):
        return (torch.rand_like(tensor) + tensor * 255.0) / 256.0

    def __repr__(self):
        return self.__class__.__name__ + "()"


class LogitTransform:
    def __call__(self, X):
        assert X.min() >= 0.0 and X.max() <= 1.0
        return torch.logit(LOGIT_LAMBDA + (1 - 2 * LOGIT_LAMBDA) * X)

    def __repr__(self):
        return self.__class__.__name__ + f"(lmd={LOGIT_LAMBDA})"


class CropCelebA64(object):
    # This class is borrowed from: https://github.com/NVlabs/NVAE
    """This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """

    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SplitDataset:

    """Wrapper for cross-validated dataset loader"""

    def __init__(
        self,
        name: str,
        root_seed: int,
        img_dim: int = 32,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
        params: Dict[str, Any] = None,
    ):
        self._name = name
        self._img_dim = img_dim
        self._batch_size = batch_size
        self._cuda = torch.cuda.is_available()
        self._device = device or torch.device("cuda" if self._cuda else "cpu")
        self._params = params or {}
        self._seed_gen = SeedGenerator(root_seed)

        self.load_dataset(name)

    def load_dataset(self, name: str) -> None:
        assert name in KNOWN_DATASETS
        if name == "BinarizedMNIST":
            return self._load_binarized_mnist()

        root = os.path.join(os.getenv("DL_DATASETS_ROOT", "/tmp"), name)
        dset = getattr(datasets, name)
        dset_idxs = dataset_with_indices(dset)
        transform = self.get_transforms(name)
        print(f"Using transforms: {transform}")
        dset_args = dict(download=True, transform=transform)
        if name == "CelebA":
            self._data = dset_idxs(root, split="train", **dset_args)
            self._valdata = dset_idxs(root, split="valid", **dset_args)
            self._testdata = dset_idxs(root, split="test", **dset_args)
        else:
            self._data = dset_idxs(root, train=True, **dset_args)
            self._testdata = dset_idxs(root, train=False, **dset_args)

    def _load_binarized_mnist(self):
        name = "MNIST"
        root = os.path.join(os.getenv("DL_DATASETS_ROOT", "/tmp"), name)
        dset = getattr(datasets, name)
        dset_idxs = dataset_with_indices(dset)

        transform = self.get_transforms("BinarizedMNIST")
        dset_args = dict(download=True, transform=transform)
        self._data = dset_idxs(root, train=True, **dset_args)
        self._testdata = dset_idxs(root, train=False, **dset_args)

    def get_transforms(self, name: str) -> List[Any]:
        """Return transforms for different datasets"""
        enabled = lambda k: k in self._params and self._params[k]
        if name == "BinarizedMNIST":
            trans = [transforms.Pad(padding=2), transforms.ToTensor()]
            if enabled("binarize"):
                trans.append(Binarize())
        elif name == "CIFAR10":
            trans = [
                transforms.Resize((self._img_dim, self._img_dim)),
            ]
            trans.append(transforms.ToTensor())
            if enabled("dequantize"):
                trans.append(UniformDequantization())
            if enabled("logits"):
                trans.append(LogitTransform())
        elif name == "CelebA":
            trans = [
                CropCelebA64(),
            ]
            if enabled("no_resize"):
                pass
            elif enabled("resize_64"):
                trans.append(transforms.Resize((64, 64)))
            else:
                trans.append(transforms.Resize((self._img_dim, self._img_dim)))

            trans.append(transforms.ToTensor())

            if enabled("dequantize"):
                trans.append(UniformDequantization())
            if enabled("logits"):
                trans.append(LogitTransform())
        else:
            print(
                f"Warning: using generic transforms for dataset: {name}",
                file=sys.stderr,
            )
            trans = [
                transforms.Resize((self._img_dim, self._img_dim)),
                transforms.ToTensor(),
            ]
        return transforms.Compose(trans)

    def get_item_by_index(self, index: int):
        """Return an item by its index in the full dataset"""
        if isinstance(self._data, Subset):
            local_idx = torch.nonzero(self._data.indices == index)
            assert not len(local_idx) == 0
            local_idx = local_idx.item()
            x, y, idx = self._data[local_idx]
        else:
            x, y, idx = self._data[index]
        assert idx == index
        return (x, y, idx)

    @property
    def name(self) -> str:
        return self._name

    @property
    def channels(self) -> int:
        if self._name in ["BinarizedMNIST"]:
            return 1
        if self._name in ["CIFAR10", "CelebA"]:
            return 3
        raise ValueError(f"Unknown dataset: {self._name}")

    def create_splits_cv(
        self, run: int, folds: int = 10
    ) -> Iterable[Tuple[DataLoader, DataLoader]]:
        N = len(self._data)

        m = N // folds
        rem = N % folds
        fold_sizes = [m + 1 if i < rem else m for i in range(folds)]

        start = 0
        perm = torch.tensor(self._seed_gen.get_permutation(N, run))
        for i in range(folds):
            out_idxs = perm[start : start + fold_sizes[i]]
            in_idxs = torch.tensor([v for v in perm if not v in out_idxs])
            start += fold_sizes[i]

            in_loader = self.create_subset_loader(in_idxs)
            out_loader = self.create_subset_loader(out_idxs)
            yield in_loader, out_loader

    def create_full(self) -> DataLoader:
        N = len(self._data)
        all_indices = torch.arange(N)
        loader = self.create_subset_loader(all_indices)
        return loader

    def create_full_test(self) -> DataLoader:
        N = len(self._testdata)
        all_indices = torch.arange(N)
        subset = Subset(self._testdata, all_indices)
        loader_kwargs = dict(batch_size=self._batch_size, shuffle=True)
        loader_kwargs |= dict(num_workers=4) if self._cuda else {}
        loader = DataLoader(subset, **loader_kwargs)
        return loader

    def create_subset_loader(self, indices: torch.Tensor):
        subset = Subset(self._data, indices)
        loader_kwargs = dict(batch_size=self._batch_size, shuffle=True)
        loader_kwargs |= dict(num_workers=4) if self._cuda else {}
        loader = DataLoader(subset, **loader_kwargs)
        return loader
