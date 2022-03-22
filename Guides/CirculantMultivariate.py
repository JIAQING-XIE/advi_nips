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

class CirculantMultivariateNorm(AutoContinuous):
    """ 
    Circulant matrix is a special form of toeplitz. 
    For example, it looks like this:
    [[ A, B, C, D, E],
     [ E, A, B, C, D],
     [ D, E, A, B, C],
     [ C, D, E, A, B],
     [ B, C, D, E, A],
    ]]
    here we only use scale vector to construct a dim * dim matrix, 
    making it positive definite and pass this matrix to the 
    multivariate gaussian model.
    YOu can also do Lowcholesky but we only set 
    a positive definite constraint here

    Usage::
        guide = CirculantMultivariateNorm(model, residual = 0.02)
        svi = SVI(model, guide, ...)
    """
    scale_constraint = constraints.softplus_positive
    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1,
                    multivariate = True, residual = 0.01):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self.multivariate = multivariate
        self.residual = residual
        self.best = False
        self.current_cov = None
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            torch.zeros((self.latent_dim,)) + self.residual, self.scale_constraint
        )

    def get_base_dist(self):
        return dist.Normal(
            torch.zeros_like(self.loc), torch.ones_like(self.loc)
        ).to_event(1)    

    def to_posdef(self, matrix):
        for i in range(0, self.latent_dim+1):
            while torch.linalg.det(matrix[:i, :i]) <= 0: # determinant of the n-th pivot positive (|A_{n}|) > 0 
                matrix = matrix + self.residual  * torch.eye(self.latent_dim)
        
        return matrix

    def get_posterior(self, *args, **kwargs):

        """
        Returns a MultivariateNormal posterior distribution.
        """
        cov = torch.zeros((self.latent_dim, self.latent_dim))
        cov[0, :] = self.scale

        scale_copy = self.scale.clone()
        
        # create a circulant matrix
        for i in range(1, self.latent_dim):
            scale_copy = torch.roll(scale_copy, shifts=1, dims=0)
            cov[i, :] = scale_copy
        
        cov = self.to_posdef(cov) # check if positive-definite constrained
        self.current_cov = cov

        return dist.MultivariateNormal(loc = self.loc, covariance_matrix=cov )

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale * self.scale_tril.diag()

    def print_best_cov(self):
        print("A new best covariance matrix:\n {}".format(self.current_cov))
