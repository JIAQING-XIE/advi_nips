from xml.dom.pulldom import PROCESSING_INSTRUCTION
import pandas as pd
import numpy as np
import torch 
import pyro
from pyro.contrib.examples.util import MNIST
import torchvision.transforms as transforms
import pyro.distributions as dist
import torch.nn as nn

class vae():
    def __init__(self):
        pass

    def split_data(self):
        raise NotImplementedError("Not being implemented")

class mnist(vae):
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

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = torch.sigmoid(self.fc21(hidden))
        return loc_img

class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Bernoulli(loc_img, validate_args=False).to_event(1),
                obs=x.reshape(-1, 784),
            )
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img


class train_module():
    def __init__(self, svi):
        self.svi = svi

    def train(self, train_loader, use_cuda=False):
        # initialize loss accumulator
        epoch_loss = 0.0
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if use_cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += self.svi.step(x)

        # return epoch loss
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        return total_epoch_loss_train

    def evaluate(self, test_loader, use_cuda=False):
        # initialize loss accumulator
        test_loss = 0.
        # compute the loss over the entire test set
        for x, _ in test_loader:
            # if on GPU put mini-batch into CUDA memory
            if use_cuda:
                x = x.cuda()
            # compute ELBO estimate and accumulate loss
            test_loss += self.svi.evaluate_loss(x)
        normalizer_test = len(test_loader.dataset)
        total_epoch_loss_test = test_loss / normalizer_test
        return total_epoch_loss_test

