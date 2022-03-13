import pandas as pd
import numpy as np
import torch 
from pyro.contrib.examples.util import MNIST
import torchvision.transforms as transforms

class VAE():
    def __init__(self):
        pass

    def split_data(self):
        raise NotImplementedError("Not being implemented")

class mnist(VAE):
    def __init__(self):
        pass

    def split_data(self, batch_size=128, use_cuda=False):
        root = './data'
        download = True
        trans = transforms.ToTensor()
        train_set = MNIST(root=root, train=True, transform=trans,
                        download=download)
        test_set = MNIST(root=root, train=False, transform=trans)

        kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
            batch_size=batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader

"""
if __name__ == "__main__":
    mnist1 = mnist()
    a, b = mnist1.split_data(use_cuda=True)
"""