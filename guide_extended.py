# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
The :mod:`pyro.infer.autoguide` module provides algorithms to automatically
generate guides from simple models, for use in :class:`~pyro.infer.svi.SVI`.
For example to generate a mean field Gaussian guide::

    def model():
        ...

    guide = AutoNormal(model)  # a mean field guide
    svi = SVI(model, guide, Adam({'lr': 1e-3}), Trace_ELBO())

Automatic guides can also be combined using :func:`pyro.poutine.block` and
:class:`AutoGuideList`.
"""
from contextlib import ExitStack

import torch
from torch import nn
from torch.distributions import biject_to

import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.util import eye_like
from pyro.nn.module import  PyroParam
from pyro.infer.autoguide.guides import AutoContinuous
from initialization import  init_to_median

class MyGuide1(AutoContinuous):
    """ 1. When the covariance matrix is symmetic, make sure it is positive definite"""
    """Then normalize the whole covariance matrix"""
    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, init_loc_fn=init_loc_fn)
    
    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params
        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            torch.full_like(self.loc, self._init_scale), self.scale_constraint
        )
        self.scale_tril = PyroParam(
            eye_like(self.loc, self.latent_dim), self.scale_tril_constraint
        )
    
    def create_symmetric(self, rand):
        """ rand is a random vector or matrix"""
        result = rand.matmul(rand.T)
        return result



class AutoMultivariateNormal(AutoContinuous):
    scale_constraint = constraints.softplus_positive
    scale_tril_constraint = constraints.unit_lower_cholesky

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params
        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            torch.full_like(self.loc, self._init_scale), self.scale_constraint
        )
        self.scale_tril = PyroParam(
            eye_like(self.loc, self.latent_dim), self.scale_tril_constraint
        )

    def get_base_dist(self):
        
        return dist.Normal(
            torch.zeros_like(self.loc), torch.ones_like(self.loc)
        ).to_event(1)
    
    def get_transform(self, *args, **kwargs):
        scale_tril = self.scale[..., None] * self.scale_tril
        return dist.transforms.LowerCholeskyAffine(self.loc, scale_tril=scale_tril)

    def get_posterior(self, *args, **kwargs):
        """
        Returns a MultivariateNormal posterior distribution.
        """
        #print(self.scale_tril)
        scale_tril = self.scale[..., None] * self.scale_tril
        return dist.MultivariateNormal(self.loc, scale_tril=scale_tril)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale * self.scale_tril.diag()

