import torch
from torch import nn
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.nn.module import  PyroParam
from pyro.infer.autoguide.guides import AutoContinuous
from initialization import  init_to_median
import math

class BlockMultivariateNorm(AutoContinuous):
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
    scale_tril_constraint = constraints.corr_cholesky_constraint
    cov_constraint = constraints.positive_definite

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.01, symmetric = True, upperbig = True,
                    cholesky = False):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self.symmetric = symmetric
        self.upperbig = upperbig
        self.cholesky = cholesky
        self.current_cov = None
        self.residual = 0.01
        self.A = None
        self.B = None
        super().__init__(model, init_loc_fn=init_loc_fn)
    
    def build_symmetric_matrix(self, random = True, residual = 0.1, matrix = None): # 0.001 for levy
        """ this function is used for making a positive definite symmetric matrix"""
        if random:
            rand = torch.rand((self.latent_dim, self.latent_dim)) 
            # semi-positive definite -> positive definite
            result = rand.matmul(rand.T) + residual * torch.eye(self.latent_dim) 
        else:
            """ which is the choice for the training process"""
            result =  matrix.matmul(matrix.T) + residual * torch.eye(matrix.size(0)) # residual, 2 * residual
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

    def to_posdef(self, matrix):
        for i in range(0, self.latent_dim+1):
            while torch.linalg.det(matrix[:i, :i]) <= 0: # determinant of the n-th pivot positive (|A_{n}|) > 0 
                matrix = matrix + self.residual  * torch.eye(matrix.shape[0])
        
        return matrix

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params

        if self.upperbig:

            A = torch.rand((math.ceil(self.latent_dim / 2), math.ceil(self.latent_dim / 2)))
            B = torch.rand((self.latent_dim - math.ceil(self.latent_dim / 2), 
                        self.latent_dim - math.ceil(self.latent_dim / 2)))

            if self.symmetric:
                A = self.build_symmetric_matrix(matrix = A, random=False)
                B = self.build_symmetric_matrix(matrix = B, random=False)
            else:
                A = self.to_posdef(A)
                B = self.to_posdef(B)


        else:
            B = torch.rand((math.ceil(self.latent_dim / 2), math.ceil(self.latent_dim / 2)))
            A = torch.rand((self.latent_dim - math.ceil(self.latent_dim / 2), 
                        self.latent_dim - math.ceil(self.latent_dim / 2)))

            if self.symmetric:
                A = self.build_symmetric_matrix(matrix = A, random=False)
                B = self.build_symmetric_matrix(matrix = B, random=False)
            else:
                A = self.to_posdef(A)
                B = self.to_posdef(B)


        self.loc = nn.Parameter(self._init_loc())

        if self.cholesky: # do LU factorization
            self.A = PyroParam(A, self.scale_tril_constraint) 
            self.B =  PyroParam(B, self.scale_tril_constraint)
        else: # just ensure that the cov is positive definite
            self.A = PyroParam(A, self.cov_constraint)
            self.B =  PyroParam(B, self.cov_constraint)

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


        scale_tril = torch.zeros((self.latent_dim, self.latent_dim))
 
        tmp = math.ceil(self.latent_dim /2)
        

        if self.upperbig:
            #self.A = self.build_symmetric_matrix(random = False, matrix = self.A)
            scale_tril[0:tmp, 0:tmp] = self.A
            #self.B = self.build_symmetric_matrix(random = False, matrix = self.B)
            scale_tril[tmp:self.latent_dim, tmp:self.latent_dim] = self.B
        else:
            #self.A = self.build_symmetric_matrix(random = False, matrix = self.A)
            scale_tril[tmp:self.latent_dim, tmp:self.latent_dim] = self.A
            #self.B = self.build_symmetric_matrix(random = False, matrix = self.B)
            scale_tril[0:tmp, 0:tmp] = self.B
        
        self.current_cov = scale_tril
        if self.cholesky:
            return dist.MultivariateNormal(self.loc, scale_tril=scale_tril)
        else:
            return dist.MultivariateNormal(self.loc, covariance_matrix=scale_tril)

    def _loc_scale(self, *args, **kwargs):
 
        return self.loc, self.current_cov.diag()

        
    def print_best_cov(self):
        print("A new best covariance matrix:\n {}".format(self.current_cov))

