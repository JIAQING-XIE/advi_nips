import torch
from torch import nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.util import eye_like
from pyro.nn.module import  PyroParam
from pyro.infer.autoguide.guides import AutoContinuous
from initialization import  init_to_median
import math

class PolyDiagNorm(AutoContinuous):
    """ This implementation of :class:`AutoContinuous` uses
     Normal distribution to construct a guide
    over the entire latent space. The different thing from DiagonalNormal is 
    that we do linear transformation (polynomial) to the scale vector, not a linear transformation to the data.  
    
    Usage::
        guide = PolyDiagNorm(model, order = 2)
        svi = SVI(model, guide, ...)
    """

    def __init__(self, model, init_loc_fn=init_to_median, epoch = 0, order = 1 ):
        self.order = order
        self.epoch = epoch
        super().__init__(model, init_loc_fn=init_loc_fn)
        self.L = None
        self.h = None
    
    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        rand_mtx = torch.rand((self.latent_dim,)) + 0.01
        # Initialize guide params
        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            rand_mtx
        )
        self.L = torch.rand((self.order, self.latent_dim))/100
        self.h = torch.rand((self.order, self.latent_dim))/100

    def get_base_dist(self):
        return dist.Normal(
            torch.zeros_like(self.loc), torch.ones_like(self.loc)
        ).to_event(1)

    def get_transform(self, *args, **kwargs):
        return dist.transforms.AffineTransform(self.loc, self.scale)

    def get_posterior(self, *args, **kwargs):
        """
        Returns a diagonal Normal posterior distribution.
        """
        order_mtx = self.scale.reshape((1,self.latent_dim))
        # we first do self-multiplication
        for i in range(1, self.order+1):
            if self.order == 1:
                break
            new_mtx = self.scale
            for k in range(2,self.order+1):
                new_mtx = new_mtx * self.scale
            new_mtx = new_mtx.reshape((1,self.latent_dim))
            order_mtx = torch.cat((order_mtx,new_mtx), axis = 0)
        # then we perform linear tranformation
        tmp = torch.zeros((1,self.latent_dim))
        for i in range(0, self.order):
            #sprint(order_mtx[i,:])
            tmp += 0.1**self.order * order_mtx[i,:] + 1
        return dist.Normal(self.loc,  tmp ).to_event(1)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale