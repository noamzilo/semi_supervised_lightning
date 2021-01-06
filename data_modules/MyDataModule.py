from typing import Any
import torch
from pytorch_lightning import LightningDataModule
from Testing.Research.datasets.ClustersDataset import ClustersDataset
from torch.utils.data import DataLoader
from Testing.Research.datasets.augmentation.AddGaussianNoise import AddGaussianNoise
from Testing.Research.datasets.augmentation.NormalizeTensor import NormalizeTensor
from torchvision.transforms import transforms


class MyDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_dims = None
        self.vocab_size = 0

        self._config = config
        self._default_dataset, self._train_dataset, self._val_dataset, self._test_dataset = None, None, None, None

    def prepare_data(self):
        # called only on 1 GPU

        # download_dataset()
        # tokenize()
        # build_vocab()

        if self._config.dataset == "toy":
            self._default_dataset, self._train_dataset, self._val_dataset, self._test_dataset = \
                ClustersDataset.clusters_dataset_by_config()
        # elif self._config.dataset == "mnist":

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        batch = batch.cuda()
        return batch

    def setup(self, stage):
        # transform = transforms.Compose([transforms.ToTensor()])
        # mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        # mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transform)
        #
        # # train/val split
        # mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
        #
        # # assign to use in dataloaders
        # self.train_dataset = mnist_train
        # self.val_dataset = mnist_val
        # self.test_dataset = mnist_test
        pass

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
        )
        return train_dataloader

    def val_dataloader(self):
        # transforms = ...
        val_dataloader = DataLoader(
            self._val_dataset,
            batch_size=self._config.batch_size,
            shuffle=False)
        return val_dataloader

    def test_dataloader(self):
        # transforms = ...
        test_dataloader = DataLoader(self._test_dataset, batch_size=self._config.batch_size, shuffle=False)
        return test_dataloader
