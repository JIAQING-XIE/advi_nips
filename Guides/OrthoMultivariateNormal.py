import torch
from torch import nn

import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.util import eye_like
from pyro.nn.module import  PyroParam
from pyro.infer.autoguide.guides import AutoContinuous
from initialization import  init_to_median

class OrthoMultiNorm(AutoContinuous):
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

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1, diagonal = True):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self.diagonal = diagonal
        super().__init__(model, init_loc_fn=init_loc_fn)
    
    def build_symmetric_matrix(self, random = True, residual = 0.1, matrix = None):
        """ this function is used for making a positive definite symmetric matrix"""
        if random:
            rand = torch.rand((self.latent_dim, self.latent_dim)) 
            # semi-positive definite -> positive definite
            result = rand.matmul(rand.T) + residual * torch.eye(self.latent_dim) 
        else:
            """ which is the choice for the training process"""
            result = matrix.matmul(matrix.T) + residual * torch.eye(self.latent_dim)
        assert torch.det(result) > 0, "please provide a higher residual"
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
        matrix = self.build_symmetric_matrix()
        matrix = self.to_diagonal(matrix) if self.diagonal else matrix
        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            torch.full_like(self.loc, self._init_scale), self.scale_constraint
        )
        # no more choleskyaffine requirement
        self.scale_tril = PyroParam(
            matrix
        )

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
        scale_tril = self.scale[..., None] * self.scale_tril
        scale_tril = self.build_symmetric_matrix(random = False, matrix = scale_tril)
        scale_tril = self.to_diagonal(scale_tril) if self.diagonal else scale_tril
        print(scale_tril)
        return dist.MultivariateNormal(self.loc, scale_tril=scale_tril)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale * self.scale_tril.diag()
