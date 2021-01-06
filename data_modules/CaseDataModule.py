import os
from pytorch_lightning import LightningDataModule
import torchvision.datasets as datasets
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from Testing.Research.config.paths import mnist_data_download_folder
from Testing.Research.datasets.real_cases.CaseDataset import CaseDataset
from typing import Optional


class CaseDataModule(LightningDataModule):
    def __init__(self, config, path):
        # TODO maybe pass the full history dumps loader? also need an adapter to train over multiple cases w/o recreating the instance or stopping the training.
        super().__init__()
        self._config = config
        self._path = path

        self._train_dataset: Optional[Subset] = None
        self._val_dataset: Optional[Subset] = None
        self._test_dataset: Optional[Subset] = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        # transform
        transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = CaseDataset(self._path)

        train_size = self._config.train_size * len(full_dataset)
        val_size = self._config.val_size * len(full_dataset)
        test_size = len(full_dataset) - train_size - val_size
        train, val, test = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

        # assign to use in dataloaders
        self._train_dataset = train
        self._val_dataset = val
        self._test_dataset = test

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._config.batch_size, num_workers=self._config.num_workers)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self._config.batch_size, num_workers=self._config.num_workers)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self._config.batch_size, num_workers=self._config.num_workers)
