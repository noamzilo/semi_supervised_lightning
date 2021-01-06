class NormalizeTensor(object):
    def __init__(self, mean=0., gain=0.5):
        self._mean = mean
        self._gain = gain

    def __call__(self, tensor):
        normalized = (tensor - self._mean) / self._gain
        return normalized

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self._mean}, gain={self._gain})'
