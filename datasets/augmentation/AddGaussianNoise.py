import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., factor=0.1):
        self._std = std
        self._mean = mean
        self._factor = factor

    def __call__(self, tensor):
        noise = torch.randn(tensor.size())
        if tensor.is_cuda:
            noise = noise.to(device="cuda")
        return tensor + self._factor * noise * self._std + self._mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self._mean}, std={self._std})'
