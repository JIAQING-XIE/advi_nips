import torch
from torch import nn

import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.util import eye_like
from pyro.nn.module import  PyroParam
from pyro.infer.autoguide.guides import AutoContinuous
from initialization import  init_to_median

class LowRankNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a 
    low rank to high rank transformation.
    Multivariate Normal distribution to construct a guide
    over the entire latent space. The guide does not depend on the model's
    ``*args, **kwargs``.

    Usage::

        guide = LowRankNormal(model, rank=2)
        svi = SVI(model, guide, ...)
    """

    scale_constraint = constraints.softplus_positive

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1, rank=None, decompose = False, random = False):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {} ".format(init_scale))
        if not (rank is None or isinstance(rank, int) and rank > 0):
            raise ValueError("Expected rank > 0 but got {}".format(rank))
        self._init_scale = init_scale
        self.rank = rank
        self.decompose = decompose
        self.random = random
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params
        self.loc = nn.Parameter(self._init_loc())
        if self.rank is None:
            self.rank = int(round(self.latent_dim ** 0.5))
        self.scale = PyroParam(
            self.loc.new_full((self.latent_dim,), 0.5 ** 0.5 * self._init_scale),
            constraint=self.scale_constraint,
        )

        # U => (latent_dim, latent_dim) PyroParam
        # sigma (covariance/cov_factor) => (latent_dim, rank) PyroParam
        # V => (rank, latent_dim) PyroParam


        self.cov_factor = nn.Parameter(
            self.loc.new_empty(self.latent_dim, self.rank).normal_(
                0, 1 / self.rank ** 0.5
            )
        )
        
        self.V = PyroParam(
            torch.rand((self.rank, self.latent_dim)),
            constraint=self.scale_constraint,
        ) 
        
        #self.V = torch.rand((self.rank, self.latent_dim))

    def build_symmetric_matrix(self, random = True, residual = 0.01, matrix = None):
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

    def get_posterior(self, *args, **kwargs):
        """
        Returns a LowRankMultivariateNormal posterior distribution.
        """
        #scale = self.scale
        #cov_factor = self.cov_factor * scale.unsqueeze(-1)
        #print(scale.unsqueeze(-1))
        #cov_diag = scale * scale

        full_cov = self.cov_factor.matmul(self.V)
        full_cov = full_cov + self.scale * torch.eye(self.latent_dim)
        full_cov = self.build_symmetric_matrix(matrix = full_cov, random=False)
        #print(full_cov)
        return dist.MultivariateNormal(self.loc, full_cov)
        #return dist.LowRankMultivariateNormal(self.loc, cov_factor, cov_diag)

    def _loc_scale(self, *args, **kwargs):
        #scale = self.scale * (self.cov_factor.pow(2).sum(-1) + 1).sqrt()
        #print(scale)
        return self.loc, scale
        