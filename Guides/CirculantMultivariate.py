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
from scipy.linalg import circulant

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
    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.9,
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
        #print(-1/self.latent_dim)
        tmp = torch.arange(1,0,-1/self.latent_dim)
        tmp_clone = tmp.clone()
        #print(tmp_clone.flip(dims = [0]))
        tmp[1:] = tmp_clone.flip(dims = [0])[:self.latent_dim-1]

        self.scale = PyroParam(
            tmp, self.scale_constraint
        )
        
        self.scale = PyroParam(
            torch.zeros((self.latent_dim,)) + 0.01* self.residual, self.scale_constraint
        )
        
    def get_base_dist(self):
        return dist.Normal(
            torch.zeros_like(self.loc), torch.ones_like(self.loc)
        ).to_event(1)    

    def to_posdef(self, matrix):
        #print()
        #for i in range(0, self.latent_dim+1):
        #while torch.linalg.det(matrix[:self.latent_dim, :self.latent_dim]) <= 0: # determinant of the n-th pivot positive (|A_{n}|) > 0 
            #print(self.scale * torch.eye(self.latent_dim))
        if matrix[0][self.latent_dim-1] > matrix[0][0]:
            matrix = matrix.fill_diagonal_((matrix[0][self.latent_dim-1] + self.residual).item() + 1)
        else:
            matrix = matrix.fill_diagonal_((matrix[0][0] + self.residual).item() + 1)


        return matrix

    def get_posterior(self, *args, **kwargs):

        """
        Returns a MultivariateNormal posterior distribution.
        """
        
        scale_copy = self.scale
        #scale_copy[2:(self.latent_dim+1)] = 0


        cov = torch.Tensor(circulant(scale_copy.detach().numpy()))
        cov = self.to_posdef(cov) # check if positive-definite constrained

        #for i in range(self.latent_dim):
        #    print(i)
        #    print(torch.linalg.det(cov[:i, :i]))

        #print(self.scale)
        self.current_cov = cov
        return dist.MultivariateNormal(loc = self.loc + self.scale, covariance_matrix= cov )

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale * self.current_cov.diag()

    def print_best_cov(self):
        print("A new best covariance matrix:\n {}".format(self.current_cov))
