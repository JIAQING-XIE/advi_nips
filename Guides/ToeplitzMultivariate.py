import torch
from torch import nn
from scipy.linalg import toeplitz
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.distributions.util import eye_like
from pyro.nn.module import  PyroParam
from pyro.infer.autoguide.guides import AutoContinuous
from initialization import  init_to_median

class ToeplitzMultivariateNorm(AutoContinuous):
    """ 
    We only use two vectors A as the first row and B as the
    first row then using the function toeplitz. So there have 
    two cases, 1) A = B, 2) A != B


    Usage::
        guide = ToeplitzMultivariateNorm(model, extra = True)
        svi = SVI(model, guide, ...)
    """
    scale_constraint = constraints.softplus_positive
    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1,
                    multivariate = True, extra = False, residual = 0.01):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self.multivariate = multivariate
        self.residual = residual
        self.extra = extra
        self.current_cov = None
        super().__init__(model, init_loc_fn=init_loc_fn)
    

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            torch.zeros((self.latent_dim,)) + self.residual, self.scale_constraint
        )
        self.scale_2 = None
        if self.extra:
            self.scale_2 = PyroParam(
            torch.zeros((self.latent_dim,)) + 0.5 * self.residual, self.scale_constraint
        )

    def get_base_dist(self):
        return dist.Normal(
            torch.zeros_like(self.loc), torch.ones_like(self.loc)
        ).to_event(1)
    

    def to_posdef(self, matrix):
        for i in range(0, self.latent_dim+1):
            while torch.linalg.det(matrix[:i, :i]) <= 0:
                matrix = matrix + self.residual  * torch.eye(self.latent_dim)
        
        return matrix

    def get_posterior(self, *args, **kwargs):

        """
        Returns a MultivariateNormal posterior distribution.
        """
        cov = torch.zeros((self.latent_dim, self.latent_dim))
        cov[:, 0] = self.scale_2 if self.extra else self.scale
        cov[0, :] = self.scale
        
        """realizing toeplitz"""
        for i in range(1, self.latent_dim):
            for j in range(1, self.latent_dim):
                cov[i, j] = cov[i-1,j-1]
        
        cov = self.to_posdef(cov)
        self.current_cov = cov
        return dist.MultivariateNormal(loc = self.loc, covariance_matrix = cov)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale * self.scale_tril.diag()

    def print_best_cov(self):
        print("A new best covariance matrix:\n {}".format(self.current_cov))
