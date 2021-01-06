import os
from pytorch_lightning import LightningDataModule
import torchvision.datasets as datasets
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
from Testing.Research.config.paths import mnist_data_download_folder


class EcgDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self._config = config

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        self._transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # this is based on the mean and std of mnist
        ])

    def prepare_data(self):
        pass

    def setup(self, stage):
        # transform
        transform = transforms.Compose([transforms.ToTensor()])
        ecg_train_full = datasets.MNIST(mnist_data_download_folder, train=True, download=False, transform=self._transforms)
        ecg_test = datasets.MNIST(mnist_data_download_folder, train=False, download=False, transform=self._transforms)

        # train/val split
        train_size = int(self._config.train_size /
                         (self._config.train_size + self._config.val_size) * len(ecg_train_full))
        val_size = len(ecg_train_full) - train_size
        ecg_train, ecg_val = torch.utils.data.random_split(ecg_train_full, [train_size, val_size])

        # assign to use in dataloaders
        self._train_dataset = ecg_train
        self._val_dataset = ecg_val
        self._test_dataset = ecg_test

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._config.batch_size, num_workers=self._config.num_workers)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self._config.batch_size, num_workers=self._config.num_workers)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self._config.batch_size, num_workers=self._config.num_workers)
