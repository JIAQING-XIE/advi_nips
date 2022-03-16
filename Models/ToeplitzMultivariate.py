from audioop import mul
from random import random
import torch
import math
from torch import nn
import scipy
from scipy.linalg import toeplitz
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.util import eye_like
from pyro.nn.module import  PyroParam
from pyro.infer.autoguide.guides import AutoContinuous
from initialization import  init_to_median

class ToeplitzMultivariateNorm(AutoContinuous):
    """ 
    We only use two vectors A as the first row and B as the
    first row then using the function toeplitz


    Usage::
        guide = ToeplitzMultivariateNorm(model, first_ele = "mean")
        svi = SVI(model, guide, ...)
    """
    scale_constraint = constraints.softplus_positive
    scale_tril_constraint = constraints.corr_cholesky_constraint

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1, diagonal = True, upperbig = False,
                    multivariate = True):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self.diagonal = diagonal
        self.upperbig = upperbig
        self.multivariate = multivariate
        super().__init__(model, init_loc_fn=init_loc_fn)
    

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params
        #matrix = self.build_symmetric_matrix()
        #matrix = self.to_diagonal(matrix) if self.diagonal else matrix
        # Initialize A 
        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            torch.full_like(self.loc, self._init_scale), self.scale_constraint
        )

        # an example here:
        #a = toeplitz(self.scale.detach().numpy(), self.A.detach().numpy())
        #print(a)
        self.scale_tril = PyroParam(
            torch.zeros((self.latent_dim, self.latent_dim)), self.scale_tril_constraint
        )

    def get_base_dist(self):
        return dist.Normal(
            torch.zeros_like(self.loc), torch.ones_like(self.loc)
        ).to_event(1)
    
    def get_transform(self, *args, **kwargs):
        """ ignored, not used... """
        scale_tril = self.scale[..., None] * self.scale_tril
        return dist.transforms.LowerCholeskyAffine(self.loc, scale_tril=scale_tril)
    
    def to_toeplitz(self):
        """ for A, B: """
        self.scale_tril[:, 0] = self.scale
        self.scale_tril[0, :] = self.scale
        print(self.scale_tril)
        
        for i in range(1, self.latent_dim):
            for j in range(1, self.latent_dim):
                self.scale_tril[i, j] = self.scale_tril[i-1,j-1]
        return self.scale_tril

    
    def get_posterior(self, *args, **kwargs):

        """
        Returns a MultivariateNormal posterior distribution.
        """
        #scale_tril = None
        #if self.multivariate:
        #scale_tril = torch.zeros((self.latent_dim, self.latent_dim))
 
        #self.scale_tril = toeplitz(self.scale.detach().numpy(), self.A.detach().numpy()))
        scale_tril = self.scale * self.to_toeplitz()
        #print(self.scale_tril)
        print(self.scale)
    
        return dist.MultivariateNormal(self.loc, scale_tril=scale_tril)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale * self.scale_tril.diag()
