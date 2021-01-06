from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import numpy as np
from Testing.Research.config.ConfigProvider import ConfigProvider


class ClustersDataset(Dataset):
    __default_dataset = None
    __default_dataset_train = None
    __default_dataset_val = None
    __default_dataset_test = None

    def __init__(self, cluster_size: int, noise_factor: float = 0, transform=None, n_clusters=2, centers_radius=4.0, data_dim=2):
        super(ClustersDataset, self).__init__()
        self._cluster_size = cluster_size
        self._noise_factor = noise_factor
        self._n_clusters = n_clusters
        self._centers_radius = centers_radius
        self._transform = transform
        self._size = self._cluster_size * self._n_clusters
        self._data_dim = data_dim

        self._create_data_clusters()
        self._combine_clusters_to_array()
        self._normalize_data()
        self._add_noise()

        # self._plot()
        pass

    @staticmethod
    def clusters_dataset_by_config(transform=None):
        if ClustersDataset.__default_dataset is not None:
            return \
                ClustersDataset.__default_dataset, \
                ClustersDataset.__default_dataset_train, \
                ClustersDataset.__default_dataset_val, \
                ClustersDataset.__default_dataset_test
        config = ConfigProvider.get_config()
        default_dataset = ClustersDataset(
            cluster_size=config.cluster_size,
            noise_factor=config.noise_factor,
            transform=transform,
            n_clusters=config.n_clusters,
            centers_radius=config.centers_radius,
        )
        
        train_size = int(config.train_size * len(default_dataset))
        val_size = int(config.val_size * len(default_dataset))
        test_size = len(default_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = \
            torch.utils.data.random_split(default_dataset, [train_size, val_size, test_size])

        ClustersDataset.__default_dataset = default_dataset
        ClustersDataset.__default_dataset_train = train_dataset
        ClustersDataset.__default_dataset_val = val_dataset
        ClustersDataset.__default_dataset_test = test_dataset

        return default_dataset, train_dataset, val_dataset, test_dataset

    def _create_data_clusters(self):
        self._clusters = [torch.zeros((self._cluster_size, self._data_dim)) for _ in range(self._n_clusters)]
        centers_radius = self._centers_radius
        for i, c in enumerate(self._clusters):
            r, x, y = 3.0, centers_radius * np.cos(i * np.pi * 2 / self._n_clusters), centers_radius * np.sin(
                i * np.pi * 2 / self._n_clusters)
            cluster_length = 1.1
            cluster_start = i * 2 * np.pi / self._n_clusters
            cluster_end = cluster_length * (i + 1) * 2 * np.pi / self._n_clusters
            cluster_inds = torch.linspace(start=cluster_start, end=cluster_end, steps=self._cluster_size,
                                          dtype=torch.float)
            c[:, 0] = r * torch.sin(cluster_inds) + y
            c[:, 1] = r * torch.cos(cluster_inds) + x

    def _plot(self):
        plt.figure()
        plt.scatter(self._noisy_values[:, 0], self._noisy_values[:, 1], s=1, color='b', label="noisy_values")
        plt.scatter(self._values[:, 0], self._values[:, 1], s=1, color='r', label="values")
        plt.legend(loc="upper left")
        plt.show()

    def _combine_clusters_to_array(self):
        size = self._size
        self._values = torch.zeros(size, 2)
        self._labels = torch.zeros(size, dtype=torch.long)
        for i, c in enumerate(self._clusters):
            self._values[i * self._cluster_size: (i + 1) * self._cluster_size, :] = self._clusters[i]
            self._labels[i * self._cluster_size: (i + 1) * self._cluster_size] = i

    def _add_noise(self):
        size = self._size

        mean = torch.zeros(size, 2)
        std = torch.ones(size, 2)
        noise = torch.normal(mean, std)
        self._noisy_values = torch.zeros(size, 2)
        self._noisy_values[:] = self._values
        self._noisy_values = self._noisy_values + noise * self._noise_factor

    def _normalize_data(self):
        values_min, values_max = torch.min(self._values), torch.max(self._values)
        self._values = (self._values - values_min) / (values_max - values_min)
        self._values = self._values * 2 - 1

    def __len__(self):
        return self._size  # number of samples in the dataset

    # def __getitem__(self, index):
    #     item = self._values[index, :]
    #     noisy_item = self._noisy_values[index, :]
    #     # if self._transform is not None:
    #     #     noisy_item = self._transform(item)
    #     return item, noisy_item, self._labels[index]

    # def __getitem__(self, index):
    #     item = self._values[index, :]
    #     noisy_item = self._noisy_values[index, :]
    #     # if self._transform is not None:
    #     #     noisy_item = self._transform(item)
    #     return item, self._labels[index]

    def __getitem__(self, index):
        item = self._values[index, :]
        noisy_item = self._noisy_values[index, :]
        # if self._transform is not None:
        #     noisy_item = self._transform(item)
        return (item, noisy_item), self._labels[index]

    @property
    def values(self):
        return self._values

    @property
    def noisy_values(self):
        return self._noisy_values


