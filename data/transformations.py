import torch

class Standardize(object):
    def __init__(self):
        self.mean = 0
        self.std = 0
        
    def __call__(self, img):
        return ((img - self.mean) / self.std).float()
    
    def fit(self, data, dim):
        self.mean = torch.mean(data, dim = dim, keepdim=True)
        self.std = torch.std(data, dim = dim, keepdim=True)
        
    def show(self):
        return self.mean, self.std
    
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, weight = 0.5):
        self.mean = mean
        self.std = std
        self.weight = weight

    def __call__(self, tensor):
        return (tensor + torch.randn(tensor.size()) * self.std + self.mean) * self.weight

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'