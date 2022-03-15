from audioop import mul
from random import random
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

class BlockDiagNorm(AutoContinuous):
    """ 
    A, B are square matrix but do not need to be a diagonal matrix
    a matrix C = [A 0;
                  0 B]
    should be made, where A.rowsize = A.columnsize = ceiling(self.latent_dim / 2) and 
    B.rowsize = B.columnsize = self.latent_dim - A.rowsize, or vice versa
    so the params for the BlockDiagNorm is larger than self.latent_dim ^ 2 / 2 but much
    smaller than self.latent_dim ^2 (an advantage might be)


    Usage::
        guide = BlockDiagNorm(model, upperbig = True, multivariate  = True)
        svi = SVI(model, guide, ...)
    """
    scale_constraint = constraints.softplus_positive
    scale_tril_constraint = constraints.corr_matrix

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1, diagonal = True, upperbig = False,
                    multivariate = True):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self.diagonal = diagonal
        self.upperbig = upperbig
        self.multivariate = multivariate
        self.A = None
        self.B = None
        super().__init__(model, init_loc_fn=init_loc_fn)
    
    def build_symmetric_matrix(self, random = True, residual = 0.1, matrix = None):
        """ this function is used for making a positive definite symmetric matrix"""
        if random:
            rand = torch.rand((self.latent_dim, self.latent_dim)) 
            # semi-positive definite -> positive definite
            result = rand.matmul(rand.T) + residual * torch.eye(self.latent_dim) 
        else:
            """ which is the choice for the training process"""
            result = matrix.matmul(matrix.T) + residual * torch.eye(matrix.size(0))
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
        if self.upperbig:
            A = torch.rand((math.ceil(self.latent_dim / 2), math.ceil(self.latent_dim / 2)))
            A = self.build_symmetric_matrix(matrix = A, random=False)
            B = torch.rand((self.latent_dim - math.ceil(self.latent_dim / 2), 
                        self.latent_dim - math.ceil(self.latent_dim / 2)))
            B = self.build_symmetric_matrix(matrix = B, random=False)
        else:
            B = torch.rand((math.ceil(self.latent_dim / 2), math.ceil(self.latent_dim / 2)))
            A = torch.rand((self.latent_dim - math.ceil(self.latent_dim / 2), 
                        self.latent_dim - math.ceil(self.latent_dim / 2)))
        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            torch.full_like(self.loc, self._init_scale), self.scale_constraint
        )
        # no more choleskyaffine requirement
        self.A = PyroParam(
            A, self.scale_tril_constraint
        )
        self.B =  PyroParam(
            B, self.scale_tril_constraint
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
        #scale_tril = None
        #if self.multivariate:
        scale_tril = torch.zeros((self.latent_dim, self.latent_dim))
 
        tmp = math.ceil(self.latent_dim /2)
        
        
        #print(scale_tril[0:tmp, 0:tmp])
        scale_tril[0:tmp, 0:tmp] = self.A
        scale_tril[self.latent_dim-tmp+1:self.latent_dim, self.latent_dim-tmp+1:self.latent_dim] = self.B
        #p
        #scale_tril = self.scale[..., None] * self.scale_tril
        scale_tril = self.scale[..., None] * scale_tril
        
        #scale_tril = self.build_symmetric_matrix(random = False, matrix = scale_tril)
        #scale_tril = self.to_diagonal(scale_tril) if self.diagonal else scale_tril
        print(scale_tril)
        
        return dist.MultivariateNormal(self.loc, scale_tril=scale_tril)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale * self.scale_tril.diag()
