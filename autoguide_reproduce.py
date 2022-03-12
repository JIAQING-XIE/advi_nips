import functools
import operator
from types import SimpleNamespace
import warnings
import weakref
from contextlib import ExitStack

import torch
from torch import nn
from torch.distributions import biject_to

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.transforms import affine_autoregressive, iterated
from pyro.distributions.util import eye_like, sum_rightmost
from pyro.infer.enum import config_enumerate
from pyro.nn.module import PyroModule, PyroParam
from pyro.ops.hessian import hessian
from pyro.ops.tensor_utils import periodic_repeat
from pyro.poutine.util import site_is_subsample

from initialization import InitMessenger, init_to_feasible, init_to_mredian
from utils import _product, deep_getattr, deep_setattr, helpful_support_errors

def prototype_hide_fn(msg):
    # Record only stochastic sites in the prototype_trace.
    return msg["type"] != "sample" or msg["is_observed"] or site_is_subsample(msg)


class AutoGuide(PyroModule):
    """
    Base class for automatic guides.
    Derived classes must implement the :meth:`forward` method, with the
    same ``*args, **kwargs`` as the base ``model``.
    Auto guides can be used individually or combined in an
    :class:`AutoGuideList` object.
    :param callable model: A pyro model.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    def __init__(self, model, *, create_plates=None):
        super().__init__(name=type(self).__name__)
        self.master = None
        # Do not register model as submodule
        self._model = (model,)
        self.create_plates = create_plates
        self.prototype_trace = None
        self._prototype_frames = {}

    @property
    def model(self):
        return self._model[0]

    def __getstate__(self):
        # Do not pickle weakrefs.
        self._model = None
        self.master = None
        return getattr(super(), "__getstate__", self.__dict__.copy)()

    def __setstate__(self, state):
        getattr(super(), "__setstate__", self.__dict__.update)(state)
        assert self.master is None
        master_ref = weakref.ref(self)
        for _, mod in self.named_modules():
            if mod is not self and isinstance(mod, AutoGuide):
                mod._update_master(master_ref)

    def _update_master(self, master_ref):
        self.master = master_ref
        for _, mod in self.named_modules():
            if mod is not self and isinstance(mod, AutoGuide):
                mod._update_master(master_ref)

    def call(self, *args, **kwargs):
        """
        Method that calls :meth:`forward` and returns parameter values of the
        guide as a `tuple` instead of a `dict`, which is a requirement for
        JIT tracing. Unlike :meth:`forward`, this method can be traced by
        :func:`torch.jit.trace_module`.
        .. warning::
            This method may be removed once PyTorch JIT tracer starts accepting
            `dict` as valid return types. 
        """
        result = self(*args, **kwargs)
        return tuple(v for _, v in sorted(result.items()))

    def sample_latent(*args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        pass

    def __setattr__(self, name, value):
        if isinstance(value, AutoGuide):
            master_ref = weakref.ref(self) if self.master is None else self.master
            value._update_master(master_ref)
        super().__setattr__(name, value)

    def _create_plates(self, *args, **kwargs):
        if self.master is None:
            if self.create_plates is None:
                self.plates = {}
            else:
                plates = self.create_plates(*args, **kwargs)
                if isinstance(plates, pyro.plate):
                    plates = [plates]
                assert all(
                    isinstance(p, pyro.plate) for p in plates
                ), "create_plates() returned a non-plate"
                self.plates = {p.name: p for p in plates}
            for name, frame in sorted(self._prototype_frames.items()):
                if name not in self.plates:
                    full_size = getattr(frame, "full_size", frame.size)
                    self.plates[name] = pyro.plate(
                        name, full_size, dim=frame.dim, subsample_size=frame.size
                    )
        else:
            assert (
                self.create_plates is None
            ), "Cannot pass create_plates() to non-master guide"
            self.plates = self.master().plates
        return self.plates

    _prototype_hide_fn = staticmethod(prototype_hide_fn)

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        model = poutine.block(self.model, self._prototype_hide_fn)
        self.prototype_trace = poutine.block(poutine.trace(model).get_trace)(
            *args, **kwargs
        )
        if self.master is not None:
            self.master()._check_prototype(self.prototype_trace)

        self._prototype_frames = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            print(site)
            for frame in site["cond_indep_stack"]:
                if frame.vectorized:
                    self._prototype_frames[frame.name] = frame
                else:
                    raise NotImplementedError(
                        "AutoGuide does not support sequential pyro.plate"
                    )

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.
        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        raise NotImplementedError


class AutoNormal(AutoGuide):
    """
    Usage::
        guide = AutoNormal(model)
        svi = SVI(model, guide, ...)
    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    scale_constraint = constraints.softplus_positive # constraint: (0, inf) <==>  0 < constraint < inf

    def __init__(self, model, *, init_loc_fn=init_to_feasible, init_scale=0.1, create_plates=None):
        self.init_loc_fn = init_loc_fn # loc is the mean and scale is the stddev

        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale

        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self._event_dims = {}
        self.locs = PyroModule()
        self.scales = PyroModule()

        # Initialize guide params
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect unconstrained event_dims, which may differ from constrained event_dims.
            with helpful_support_errors(site):
                init_loc = (
                    biject_to(site["fn"].support).inv(site["value"].detach()).detach()
                )

            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim

            # If subsampling, repeat init_value to full size.
            for frame in site["cond_indep_stack"]:
                full_size = getattr(frame, "full_size", frame.size)
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    init_loc = periodic_repeat(init_loc, full_size, dim).contiguous()
            #init_scale = torch.full_like(init_loc, self._init_scale)
            init_scale = torch.full_like(init_loc, self._init_scale)

            deep_setattr(
                self.locs, name, PyroParam(init_loc, constraints.real, event_dim)
            )
            deep_setattr(
                self.scales,
                name,
                PyroParam(init_scale, self.scale_constraint, event_dim),
            )

    def _get_loc_and_scale(self, name):
        site_loc = deep_getattr(self.locs, name)
        site_scale = deep_getattr(self.scales, name)
        return site_loc, site_scale

    def forward(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.
        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__`.
        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            transform = biject_to(site["fn"].support)
            print(site["cond_indep_stack"])
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])

                site_loc, site_scale = self._get_loc_and_scale(name)
                unconstrained_latent = pyro.sample(
                    name + "_unconstrained",
                    dist.Normal(
                        site_loc,
                        site_scale,
                    ).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True},
                )

                value = transform(unconstrained_latent)
                if poutine.get_mask() is False:
                    log_density = 0.0
                else:
                    log_density = transform.inv.log_abs_det_jacobian(
                        value,
                        unconstrained_latent,
                    )
                    log_density = sum_rightmost(
                        log_density,
                        log_density.dim() - value.dim() + site["fn"].event_dim,
                    )
                delta_dist = dist.Delta(
                    value,
                    log_density=log_density,
                    event_dim=site["fn"].event_dim,
                )

                result[name] = pyro.sample(name, delta_dist)

        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.
        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        medians = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, _ = self._get_loc_and_scale(name)
            median = biject_to(site["fn"].support)(site_loc)
            if median is site_loc:
                median = median.clone()
            medians[name] = median

        return medians


class AutoContinuous(AutoGuide):
    """
    Base class for implementations of continuous-valued Automatic
    Differentiation Variational Inference [1].
    This uses :mod:`torch.distributions.transforms` to transform each
    constrained latent variable to an unconstrained space, then concatenate all
    variables into a single unconstrained latent variable.  Each derived class
    implements a :meth:`get_posterior` method returning a distribution over
    this single unconstrained latent variable.
    Assumes model structure and latent dimension are fixed, and all latent
    variables are continuous.
    :param callable model: a Pyro model
    Reference:
    [1] `Automatic Differentiation Variational Inference`,
        Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, David M.
        Blei
    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    """

    def __init__(self, model, init_loc_fn=init_to_median):
        model = InitMessenger(init_loc_fn)(model)
        super().__init__(model)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        self._unconstrained_shapes = {}
        self._cond_indep_stacks = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect the shapes of unconstrained values.
            # These may differ from the shapes of constrained values.
            with helpful_support_errors(site):
                self._unconstrained_shapes[name] = (
                    biject_to(site["fn"].support).inv(site["value"]).shape
                )

            # Collect independence contexts.
            self._cond_indep_stacks[name] = site["cond_indep_stack"]

        self.latent_dim = sum(
            _product(shape) for shape in self._unconstrained_shapes.values()
        )
        if self.latent_dim == 0:
            raise RuntimeError(
                "{} found no latent variables; Use an empty guide instead".format(
                    type(self).__name__
                )
            )
    def _init_loc(self):
        """
        Creates an initial latent vector using a per-site init function.
        """
        parts = []
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            constrained_value = site["value"].detach()
            unconstrained_value = biject_to(site["fn"].support).inv(constrained_value)
            parts.append(unconstrained_value.reshape(-1))
        latent = torch.cat(parts)
        assert latent.size() == (self.latent_dim,)
        return latent

    def get_base_dist(self):
        """
        Returns the base distribution of the posterior when reparameterized
        as a :class:`~pyro.distributions.TransformedDistribution`. This
        should not depend on the model's `*args, **kwargs`.
        .. code-block:: python
          posterior = TransformedDistribution(self.get_base_dist(), self.get_transform(*args, **kwargs))
        :return: :class:`~pyro.distributions.TorchDistribution` instance representing the base distribution.
        """
        raise NotImplementedError

    def get_transform(self, *args, **kwargs):
        """
        Returns the transform applied to the base distribution when the posterior
        is reparameterized as a :class:`~pyro.distributions.TransformedDistribution`.
        This may depend on the model's `*args, **kwargs`.
        .. code-block:: python
          posterior = TransformedDistribution(self.get_base_dist(), self.get_transform(*args, **kwargs))
        :return: a :class:`~torch.distributions.Transform` instance.
        """
        raise NotImplementedError

    def get_posterior(self, *args, **kwargs):
        """
        Returns the posterior distribution.
        """
        base_dist = self.get_base_dist()
        transform = self.get_transform(*args, **kwargs)
        return dist.TransformedDistribution(base_dist, transform)

    def sample_latent(self, *args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        pos_dist = self.get_posterior(*args, **kwargs)
        return pyro.sample(
            "_{}_latent".format(self._pyro_name), pos_dist, infer={"is_auxiliary": True}
        )

    def _unpack_latent(self, latent):
        """
        Unpacks a packed latent tensor, iterating over tuples of the form::
            (site, unconstrained_value)
        """
        batch_shape = latent.shape[
            :-1
        ]  # for plates outside of _setup_prototype, e.g. parallel particles
        pos = 0
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            constrained_shape = site["value"].shape
            unconstrained_shape = self._unconstrained_shapes[name]
            size = _product(unconstrained_shape)
            event_dim = (
                site["fn"].event_dim + len(unconstrained_shape) - len(constrained_shape)
            )
            unconstrained_shape = torch.broadcast_shapes(
                unconstrained_shape, batch_shape + (1,) * event_dim
            )
            unconstrained_value = latent[..., pos : pos + size].view(
                unconstrained_shape
            )
            yield site, unconstrained_value
            pos += size
        if not torch._C._get_tracing_state():
            assert pos == latent.size(-1)

    def forward(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.
        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__`.
        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        latent = self.sample_latent(*args, **kwargs)
        plates = self._create_plates(*args, **kwargs)

        # unpack continuous latent samples
        result = {}
        for site, unconstrained_value in self._unpack_latent(latent):
            name = site["name"]
            transform = biject_to(site["fn"].support)
            value = transform(unconstrained_value)
            if poutine.get_mask() is False:
                log_density = 0.0
            else:
                log_density = transform.inv.log_abs_det_jacobian(
                    value,
                    unconstrained_value,
                )
                log_density = sum_rightmost(
                    log_density,
                    log_density.dim() - value.dim() + site["fn"].event_dim,
                )
            delta_dist = dist.Delta(
                value,
                log_density=log_density,
                event_dim=site["fn"].event_dim,
            )

            with ExitStack() as stack:
                for frame in self._cond_indep_stacks[name]:
                    stack.enter_context(plates[frame.name])
                result[name] = pyro.sample(name, delta_dist)

        return result

    def _loc_scale(self, *args, **kwargs):
        """
        :returns: a tuple ``(loc, scale)`` used by :meth:`median` and
            :meth:`quantiles`
        """
        raise NotImplementedError

    @torch.no_grad()
    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.
        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        loc, _ = self._loc_scale(*args, **kwargs)
        loc = loc.detach()
        return {
            site["name"]: biject_to(site["fn"].support)(unconstrained_value)
            for site, unconstrained_value in self._unpack_latent(loc)
        }

    @torch.no_grad()
    def quantiles(self, quantiles, *args, **kwargs):
        """
        Returns posterior quantiles each latent variable. Example::
            print(guide.quantiles([0.05, 0.5, 0.95]))
        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a tensor of quantile values.
        :rtype: dict
        """
        loc, scale = self._loc_scale(*args, **kwargs)
        quantiles = torch.tensor(
            quantiles, dtype=loc.dtype, device=loc.device
        ).unsqueeze(-1)
        latents = dist.Normal(loc, scale).icdf(quantiles)
        result = {}
        for latent in latents:
            for site, unconstrained_value in self._unpack_latent(latent):
                result.setdefault(site["name"], []).append(
                    biject_to(site["fn"].support)(unconstrained_value)
                )
        result = {k: torch.stack(v) for k, v in result.items()}
        return result

class AutoMultivariateNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Cholesky
    factorization of a Multivariate Normal distribution to construct a guide
    over the entire latent space. The guide does not depend on the model's
    ``*args, **kwargs``.
    Usage::
        guide = AutoMultivariateNormal(model)
        svi = SVI(model, guide, ...)
    By default the mean vector is initialized by ``init_loc_fn()`` and the
    Cholesky factor is initialized to the identity times a small factor.
    :param callable model: A generative model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    """

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
        scale_tril = self.scale[..., None] * self.scale_tril
        return dist.MultivariateNormal(self.loc, scale_tril=scale_tril)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale * self.scale_tril.diag()

class AutoDiagonalNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.
    Usage::
        guide = AutoDiagonalNormal(model)
        svi = SVI(model, guide, ...)
    By default the mean vector is initialized to zero and the scale is
    initialized to the identity times a small factor.
    :param callable model: A generative model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    """

    scale_constraint = constraints.softplus_positive

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
            self.loc.new_full((self.latent_dim,), self._init_scale),
            self.scale_constraint,
        )

    def get_base_dist(self):
        return dist.Normal(
            torch.zeros_like(self.loc), torch.ones_like(self.loc)
        ).to_event(1)

    def get_transform(self, *args, **kwargs):
        return dist.transforms.AffineTransform(self.loc, self.scale)

    def get_posterior(self, *args, **kwargs):
        """
        Returns a diagonal Normal posterior distribution.
        """
        return dist.Normal(self.loc, self.scale).to_event(1)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale


class AutoNormalizingFlow(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Diagonal Normal
    distribution transformed via a sequence of bijective transforms
    (e.g. various :mod:`~pyro.distributions.TransformModule` subclasses)
    to construct a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.
    Usage::
        transform_init = partial(iterated, block_autoregressive,
                                 repeats=2)
        guide = AutoNormalizingFlow(model, transform_init)
        svi = SVI(model, guide, ...)
    :param callable model: a generative model
    :param init_transform_fn: a callable which when provided with the latent
        dimension returns an instance of :class:`~torch.distributions.Transform`
        , or :class:`~pyro.distributions.TransformModule` if the transform has
        trainable params.
    """

    def __init__(self, model, init_transform_fn):
        super().__init__(model, init_loc_fn=init_to_feasible)
        self._init_transform_fn = init_transform_fn
        self.transform = None
        self._prototype_tensor = torch.tensor(0.0)

    def get_base_dist(self):
        loc = self._prototype_tensor.new_zeros(1)
        scale = self._prototype_tensor.new_ones(1)
        return dist.Normal(loc, scale).expand([self.latent_dim]).to_event(1)

    def get_transform(self, *args, **kwargs):
        return self.transform

    def get_posterior(self, *args, **kwargs):
        if self.transform is None:
            self.transform = self._init_transform_fn(self.latent_dim)
            # Update prototype tensor in case transform parameters
            # device/dtype is not the same as default tensor type.
            for _, p in self.named_pyro_params():
                self._prototype_tensor = p
                break
        return super().get_posterior(*args, **kwargs)



from collections import OrderedDict, defaultdict
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Union

import torch
from torch.distributions import biject_to

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.util import eye_like, is_identically_zero
from pyro.infer.inspect import get_dependencies
from pyro.nn.module import PyroModule, PyroParam

from .guides import AutoGuide
from .initialization import InitMessenger, init_to_feasible
from .utils import deep_getattr, deep_setattr, helpful_support_errors

class AutoStructured(AutoGuide):
    """
    Structured guide whose conditional distributions are Delta, Normal,
    MultivariateNormal, or by a callable, and whose latent variables can depend
    on each other either linearly (in unconstrained space) or via shearing by a
    callable.
    Usage::
        def model(data):
            x = pyro.sample("x", dist.LogNormal(0, 1))
            with pyro.plate("plate", len(data)):
                y = pyro.sample("y", dist.Normal(0, 1))
                pyro.sample("z", dist.Normal(y, x), obs=data)
        # Either fully automatic...
        guide = AutoStructured(model)
        # ...or with specified conditional and dependency types...
        guide = AutoStructured(
            model, conditionals="normal", dependencies="linear"
        )
        # ...or with custom dependency structure and distribution types.
        guide = AutoStructured(
            model=model,
            conditionals={"x": "normal", "y": "delta"},
            dependencies={"x": {"y": "linear"}},
        )
    Once trained, this guide can be used with
    :class:`~pyro.infer.reparam.structured.StructuredReparam` to precondition a
    model for use in HMC and NUTS inference.
    .. note:: If you declare a dependency of a high-dimensional downstream
        variable on a low-dimensional upstream variable, you may want to use
        a lower learning rate for that weight, e.g.::
            def optim_config(param_name):
                config = {"lr": 0.01}
                if "deps.my_downstream.my_upstream" in param_name:
                    config["lr"] *= 0.1
                return config
            adam = pyro.optim.Adam(optim_config)
    :param callable model: A Pyro model.
    :param conditionals: Either a single distribution type or a dict mapping
        each latent variable name to a distribution type. A distribution type
        is either a string in {"delta", "normal", "mvn"} or a callable that
        returns a sample from a zero mean (or approximately centered) noise
        distribution (such callables typically call ``pyro.param()`` and
        ``pyro.sample()`` internally).
    :param dependencies: Dependency type, or a dict mapping each site name to a
        dict mapping its upstream dependencies to dependency types. If only a
        dependecy type is provided, dependency structure will be inferred. A
        dependency type is either the string "linear" or a callable that maps a
        *flattened* upstream perturbation to *flattened* downstream
        perturbation. The string "linear" is equivalent to
        ``nn.Linear(upstream.numel(), downstream.numel(), bias=False)``.
        Dependencies must not contain cycles or self-loops.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    scale_constraint = constraints.softplus_positive
    scale_tril_constraint = constraints.softplus_lower_cholesky

    def __init__(
        self,
        model,
        *,
        conditionals: Union[str, Dict[str, Union[str, Callable]]] = "mvn",
        dependencies: Union[str, Dict[str, Dict[str, Union[str, Callable]]]] = "linear",
        init_loc_fn: Callable = init_to_feasible,
        init_scale: float = 0.1,
        create_plates: Optional[Callable] = None,
    ):
        assert isinstance(conditionals, (dict, str))
        if isinstance(conditionals, dict):
            for name, fn in conditionals.items():
                assert isinstance(name, str)
                assert isinstance(fn, str) or callable(fn)
        assert isinstance(dependencies, (dict, str))
        if isinstance(dependencies, dict):
            for downstream, deps in dependencies.items():
                assert downstream in conditionals
                assert isinstance(deps, dict)
                for upstream, dep in deps.items():
                    assert upstream in conditionals
                    assert upstream != downstream
                    assert isinstance(dep, str) or callable(dep)
        self.conditionals = conditionals
        self.dependencies = dependencies

        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError(f"Expected init_scale > 0. but got {init_scale}")
        self._init_scale = init_scale
        self._original_model = (model,)
        model = InitMessenger(init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _auto_config(self, sample_sites, args, kwargs):
        # Instantiate conditionals as dictionaries.
        if not isinstance(self.conditionals, dict):
            self.conditionals = {
                name: self.conditionals for name, site in sample_sites.items()
            }

        # Instantiate dependencies as dictionaries.
        if not isinstance(self.dependencies, dict):
            model = self._original_model[0]
            meta = poutine.block(get_dependencies)(model, args, kwargs)
            # Use posterior dependency edges but with prior ordering. This
            # allows sampling of globals before locals on which they depend.
            prior_order = {name: i for i, name in enumerate(sample_sites)}
            dependencies = defaultdict(dict)
            for d, upstreams in meta["posterior_dependencies"].items():
                assert d in sample_sites
                for u, plates in upstreams.items():
                    # TODO use plates to reduce dimension of dependency.
                    if u in sample_sites:
                        if prior_order[u] > prior_order[d]:
                            dependencies[u][d] = self.dependencies
                        elif prior_order[d] > prior_order[u]:
                            dependencies[d][u] = self.dependencies
            self.dependencies = dict(dependencies)
        self._original_model = None

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self.locs = PyroModule()
        self.scales = PyroModule()
        self.scale_trils = PyroModule()
        self.conds = PyroModule()
        self.deps = PyroModule()
        self._batch_shapes = {}
        self._unconstrained_event_shapes = {}
        sample_sites = OrderedDict(self.prototype_trace.iter_stochastic_nodes())
        self._auto_config(sample_sites, args, kwargs)

        # Collect unconstrained shapes.
        init_locs = {}
        numel = {}
        for name, site in sample_sites.items():
            with helpful_support_errors(site):
                init_loc = (
                    biject_to(site["fn"].support).inv(site["value"].detach()).detach()
                )
            self._batch_shapes[name] = site["fn"].batch_shape
            self._unconstrained_event_shapes[name] = init_loc.shape[
                len(site["fn"].batch_shape) :
            ]
            numel[name] = init_loc.numel()
            init_locs[name] = init_loc.reshape(-1)

        # Initialize guide params.
        children = defaultdict(list)
        num_pending = {}
        for name, site in sample_sites.items():
            # Initialize location parameters.
            init_loc = init_locs[name]
            deep_setattr(self.locs, name, PyroParam(init_loc))

            # Initialize parameters of conditional distributions.
            conditional = self.conditionals[name]
            if callable(conditional):
                deep_setattr(self.conds, name, conditional)
            else:
                if conditional not in ("delta", "normal", "mvn"):
                    raise ValueError(f"Unsupported conditional type: {conditional}")
                if conditional in ("normal", "mvn"):
                    init_scale = torch.full_like(init_loc, self._init_scale)
                    deep_setattr(
                        self.scales, name, PyroParam(init_scale, self.scale_constraint)
                    )
                if conditional == "mvn":
                    init_scale_tril = eye_like(init_loc, init_loc.numel())
                    deep_setattr(
                        self.scale_trils,
                        name,
                        PyroParam(init_scale_tril, self.scale_tril_constraint),
                    )

            # Initialize dependencies on upstream variables.
            num_pending[name] = 0
            deps = PyroModule()
            deep_setattr(self.deps, name, deps)
            for upstream, dep in self.dependencies.get(name, {}).items():
                assert upstream in sample_sites
                children[upstream].append(name)
                num_pending[name] += 1
                if isinstance(dep, str) and dep == "linear":
                    dep = torch.nn.Linear(numel[upstream], numel[name], bias=False)
                    dep.weight.data.zero_()
                elif not callable(dep):
                    raise ValueError(
                        f"Expected either the string 'linear' or a callable, but got {dep}"
                    )
                deep_setattr(deps, upstream, dep)

        # Topologically sort sites.
        # TODO should we choose a more optimal structure?
        self._sorted_sites = []
        while num_pending:
            name, count = min(num_pending.items(), key=lambda kv: (kv[1], kv[0]))
            assert count == 0, f"cyclic dependency: {name}"
            del num_pending[name]
            for child in children[name]:
                num_pending[child] -= 1
            site = self._compress_site(sample_sites[name])
            self._sorted_sites.append((name, site))

        # Prune non-essential parts of the trace to save memory.
        for name, site in self.prototype_trace.nodes.items():
            site.clear()

    @staticmethod
    def _compress_site(site):
        # Save memory by retaining only necessary parts of the site.
        return {
            "name": site["name"],
            "type": site["type"],
            "cond_indep_stack": site["cond_indep_stack"],
            "fn": SimpleNamespace(
                support=site["fn"].support,
                event_dim=site["fn"].event_dim,
            ),
        }

    @poutine.infer_config(config_fn=_config_auxiliary)
    def get_deltas(self, save_params=None):
        deltas = {}
        aux_values = {}
        compute_density = poutine.get_mask() is not False
        for name, site in self._sorted_sites:
            if save_params is not None and name not in save_params:
                continue

            # Sample zero-mean blockwise independent Delta/Normal/MVN.
            log_density = 0.0
            loc = deep_getattr(self.locs, name)
            zero = torch.zeros_like(loc)
            conditional = self.conditionals[name]
            if callable(conditional):
                aux_value = deep_getattr(self.conds, name)()
            elif conditional == "delta":
                aux_value = zero
            elif conditional == "normal":
                aux_value = pyro.sample(
                    name + "_aux",
                    dist.Normal(zero, 1).to_event(1),
                    infer={"is_auxiliary": True},
                )
                scale = deep_getattr(self.scales, name)
                aux_value = aux_value * scale
                if compute_density:
                    log_density = (-scale.log()).expand_as(aux_value)
            elif conditional == "mvn":
                # This overparametrizes by learning (scale,scale_tril),
                # enabling faster learning of the more-global scale parameter.
                aux_value = pyro.sample(
                    name + "_aux",
                    dist.Normal(zero, 1).to_event(1),
                    infer={"is_auxiliary": True},
                )
                scale = deep_getattr(self.scales, name)
                scale_tril = deep_getattr(self.scale_trils, name)
                aux_value = aux_value @ scale_tril.T * scale
                if compute_density:
                    log_density = (
                        -scale_tril.diagonal(dim1=-2, dim2=-1).log() - scale.log()
                    ).expand_as(aux_value)
            else:
                raise ValueError(f"Unsupported conditional type: {conditional}")

            # Accumulate upstream dependencies.
            # Note: by accumulating upstream dependencies before updating the
            # aux_values dict, we encode a block-sparse structure of the
            # precision matrix; if we had instead accumulated after updating
            # aux_values, we would encode a block-sparse structure of the
            # covariance matrix.
            # Note: these shear transforms have no effect on the Jacobian
            # determinant, and can therefore be excluded from the log_density
            # computation below, even for nonlinear dep().
            deps = deep_getattr(self.deps, name)
            for upstream in self.dependencies.get(name, {}):
                dep = deep_getattr(deps, upstream)
                aux_value = aux_value + dep(aux_values[upstream])
            aux_values[name] = aux_value

            # Shift by loc and reshape.
            batch_shape = torch.broadcast_shapes(
                aux_value.shape[:-1], self._batch_shapes[name]
            )
            unconstrained = (aux_value + loc).reshape(
                batch_shape + self._unconstrained_event_shapes[name]
            )
            if not is_identically_zero(log_density):
                log_density = log_density.reshape(batch_shape + (-1,)).sum(-1)

            # Transform to constrained space.
            transform = biject_to(site["fn"].support)
            value = transform(unconstrained)
            if compute_density and conditional != "delta":
                assert transform.codomain.event_dim == site["fn"].event_dim
                log_density = log_density + transform.inv.log_abs_det_jacobian(
                    value, unconstrained
                )

            # Create a reparametrized Delta distribution.
            deltas[name] = dist.Delta(value, log_density, site["fn"].event_dim)

        return deltas

    def forward(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        deltas = self.get_deltas()
        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self._sorted_sites:
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                result[name] = pyro.sample(name, deltas[name])

        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        result = {}
        for name, site in self._sorted_sites:
            loc = deep_getattr(self.locs, name).detach()
            shape = self._batch_shapes[name] + self._unconstrained_event_shapes[name]
            loc = loc.reshape(shape)
            result[name] = biject_to(site["fn"].support)(loc)
        return result








if __name__ == "__main__":
    def model_1(data):
        m = pyro.sample("m", dist.Normal(0, 1))
        sd = pyro.sample("sd", dist.LogNormal(m, 1))
        with pyro.plate("N", len(data)):
            pyro.sample("obs", dist.Normal(m, sd), obs=data)

    data = torch.ones(10)
    cond_model = pyro.condition(model_1, data={"obs": data})

    sites = poutine.trace(cond_model).get_trace(data)
    print("sd" in sites.nodes)
    guide = AutoNormal(cond_model)
    guide._setup_prototype(data)

    model = poutine.block(cond_model, hide = ["m"])
    print("sd" in poutine.trace(cond_model).get_trace(data))
