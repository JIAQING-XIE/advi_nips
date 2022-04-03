import torch
from torch import nn

import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.util import eye_like
from pyro.nn.module import  PyroParam
from pyro.infer.autoguide.guides import AutoContinuous
from initialization import  init_to_median

class SymMultiNorm(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Cholesky
    factorization of a Multivariate Normal distribution to construct a guide
    over the entire latent space. Used full-rank assumptions and VUV^T decomposition
    The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = OrthoMultiNorm(model)
        svi = SVI(model, guide, ...)
    """

    scale_constraint = constraints.softplus_positive

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1, diagonal = False, residual = 0.01):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self.diagonal = diagonal
        self.residual = residual 
        super().__init__(model, init_loc_fn=init_loc_fn)
    
    def build_symmetric_matrix(self, random = True, matrix = None):
        """ this function is used for making a positive definite symmetric matrix"""
        if random:
            rand = torch.rand((self.latent_dim, self.latent_dim)) 
            # semi-positive definite -> positive definite
            result = self.residual * rand.matmul(rand.T) +  torch.eye(self.latent_dim) 
        else:

            """ which is the choice for the training process"""
            #print( matrix.matmul(matrix.T))
            result = self.residual * matrix.matmul(matrix.T) +  self.residual * torch.eye(self.latent_dim) # for levy
            #for other data: result = self.residual * matrix.matmul(matrix.T) + torch.eye(self.latent_dim) 
        
        #assert torch.det(result) > 0, "please provide a higher residual"
        return result
        
    def to_diagonal(self, matrix = None):
        """ check if we want our posdef symmetric matrix to be diagonal, if so then execute else quick"""
        if self.diagonal == False:
            return matrix
        else:
            assert torch.equal(matrix, matrix.T) == True, "Input should be a symmetric matrix"
            assert torch.det(matrix) > 0, "please provide a positive defnite matrix"
            # matrix = V D V^T, or matrix = V D V^-1
            D, V = torch.linalg.eig(matrix) # D is the eigen vectors and V is the reversible matrix
            return torch.diag(D).real # does not consider complex conditions

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params

        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            torch.rand((self.latent_dim,)), self.scale_constraint
        )
        self.scale = PyroParam(
            torch.zeros((self.latent_dim,)) + self.residual, self.scale_constraint
        )
        #print(self.scale)
        # no more choleskyaffine requirement

    def get_base_dist(self):
        return dist.Normal(
            torch.zeros_like(self.loc), torch.ones_like(self.loc)
        ).to_event(1)

    def get_transform(self, *args, **kwargs):
        """ ignored, not used... """
        scale_tril = self.scale[..., None] * self.scale_tril
        return dist.transforms.LowerCholeskyAffine(self.loc, scale_tril=scale_tril)
    
    def get_posterior(self, *args, **kwargs):
        
        """
        Returns a MultivariateNormal posterior distribution.
        """
        mtx = self.scale.clone().reshape((self.latent_dim,1))
        #print(self.scale)
        cov = self.build_symmetric_matrix(random = False, matrix = mtx)
        #print(cov)
        cov = self.to_diagonal(cov) if self.diagonal else cov
        #print(self.scale)
        return dist.MultivariateNormal(self.loc, covariance_matrix= cov)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale 
    