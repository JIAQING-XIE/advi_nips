from collections import defaultdict
from pprint import pprint
import functools
import math
import os
import torch
import pyro
import pyro.distributions as dist 
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from Guides.guide_list import guide_list
from pyro.infer.reparam import AutoReparam, LocScaleReparam
from pyro.infer.autoguide import AutoGuideList
from pyro.optim import ClippedAdam
from pyro.ops.special import sparse_multinomial_likelihood
import matplotlib.pyplot as plt
from pyro.contrib.examples.nextstrain import load_nextstrain_counts
from argparse import ArgumentParser
if torch.cuda.is_available():
    print("Using GPU")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("Using CPU")
smoke_test = ('CI' in os.environ)

def parse():
    parser = ArgumentParser()
    parser.add_argument('--method', type = str, help = 'autoguide methods')
    parser.add_argument('--epochs', default = 1000, type = int, help = 'number of epochs')
    parser.add_argument('--lr', default= 0.1, type=float, help = "learning rate")
    return parser.parse_args()

def model(dataset, predict=None):
    features = dataset["features"]
    counts = dataset["counts"]
    sparse_counts = dataset["sparse_counts"]
    assert features.shape[0] == counts.shape[-1]
    S, M = features.shape
    T, P, S = counts.shape
    time = torch.arange(float(T)) * dataset["time_step_days"] / 5.5
    time -= time.mean()

    # Model each region as multivariate logistic growth.
    rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))
    init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))
    with pyro.plate("mutation", M, dim=-1):
        coef = pyro.sample("coef", dist.Laplace(0, 0.5))
    with pyro.plate("strain", S, dim=-1):
        rate_loc = pyro.deterministic("rate_loc", 0.01 * coef @ features.T)
        with pyro.plate("place", P, dim=-2):
            rate = pyro.sample("rate", dist.Normal(rate_loc, rate_scale))
            init = pyro.sample("init", dist.Normal(0, init_scale))
    if predict is not None:  # Exit early during evaluation.
        probs = (init + rate * time[predict]).softmax(-1)
        return probs
    logits = (init + rate * time[:, None, None]).log_softmax(-1)

    # Observe sequences via a cheap sparse multinomial likelihood.
    t, p, s = sparse_counts["index"]
    pyro.factor(
        "obs",
        sparse_multinomial_likelihood(
            sparse_counts["total"], logits[t, p, s], sparse_counts["value"]
        )
    )

def fit_svi(model, guide, gname, lr=0.01, num_steps=2001, log_every=100, plot=True, ):
    pyro.clear_param_store()
    pyro.set_rng_seed(20211205)
    if smoke_test:
        num_steps = 2

    # Measure model and guide complexity.
    num_latents = sum(
        site["value"].numel()
        for name, site in poutine.trace(guide).get_trace(dataset).iter_stochastic_nodes()
        if not site["infer"].get("is_auxiliary")
    )
    num_params = sum(p.unconstrained().numel() for p in pyro.get_param_store().values())
    print(f"Found {num_latents} latent variables and {num_params} learnable parameters")
    
    # Save gradient norms during inference.
    series = defaultdict(list)
    def hook(g, series):
        series.append(torch.linalg.norm(g.reshape(-1), math.inf).item())
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(
            functools.partial(hook, series=series[name + " grad"])
        )

    # Train the guide.
    optim = ClippedAdam({"lr": lr, "lrd": 0.1 ** (1 / num_steps)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    num_obs = int(dataset["counts"].count_nonzero())
    for step in range(num_steps):
        loss = svi.step(dataset) / num_obs
        series["loss"].append(loss)
        median = guide.median()  # cheap for autoguides
        for name, value in median.items():
            if value.numel() == 1:
                series[name + " mean"].append(float(value))
        if step % log_every == 0:
            print(f"step {step: >4d} loss = {loss:0.6g}")

    # Plot series to assess convergence.
    if plot:
        plt.figure(figsize=(6, 6))
        for name, Y in series.items():
            if name == "loss":
                plt.plot(Y, "k--", label=name, zorder=0)
            elif name.endswith(" mean"):
                plt.plot(Y, label=name, zorder=-1)
            else:
                plt.plot(Y, label=name, alpha=0.5, lw=1, zorder=-2)
        plt.xlabel("SVI step")
        plt.title("loss, scalar parameters, and gradient norms")
        plt.yscale("log")
        plt.xscale("symlog")
        plt.xlim(0, None)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig("./Results/sars_cov2/params_loss_{}".format(gname))

def mae(true_counts, pred_probs):
    """Computes mean average error between counts and predicted probabilities."""
    pred_counts = pred_probs * true_counts.sum(-1, True)
    error = (true_counts - pred_counts).abs().sum(-1)
    total = true_counts.sum(-1).clamp(min=1)
    return (error / total).mean().item()

def evaluate(
    model, guide, num_particles=100, location="USA / Massachusetts", time=-2
):
    if smoke_test:
        num_particles = 4
    """Evaluate posterior predictive accuracy at the last fully observed time step."""
    with torch.no_grad(), poutine.mask(mask=False):  # makes computations cheaper
        with pyro.plate("particle", num_particles, dim=-3):  # vectorizes
            guide_trace = poutine.trace(guide).get_trace(dataset)
            probs = poutine.replay(model, guide_trace)(dataset, predict=time)
        probs = probs.squeeze().mean(0)  # average over Monte Carlo samples
        true_counts = dataset["counts"][time]
        # Compute global and local KL divergence.
        global_mae = mae(true_counts, probs)
        i = dataset["locations"].index(location)
        local_mae = mae(true_counts[i], probs[i])
    return {"MAE (global)": global_mae, f"MAE ({location})": local_mae}

def plot_volcano(guide, gname, num_particles=100):
    if smoke_test:
        num_particles = 4
    with torch.no_grad(), poutine.mask(mask=False):  # makes computations cheaper
        with pyro.plate("particle", num_particles, dim=-3):  # vectorizes
            trace = poutine.trace(guide).get_trace(dataset)
            trace = poutine.trace(poutine.replay(model, trace)).get_trace(dataset, -1)
            coef = trace.nodes["coef"]["value"].cpu()
    coef = coef.squeeze() * 0.01  # Scale factor as in the model.
    mean = coef.mean(0)
    std = coef.std(0)
    z_score = mean.abs() / std
    effect_size = mean.exp().numpy()
    plt.figure(figsize=(6, 3))
    plt.scatter(effect_size, z_score.numpy(), lw=0, s=5, alpha=0.5, color="darkred")
    plt.yscale("symlog")
    plt.ylim(0, None)
    plt.xlabel("$R_m/R_{wt}$")
    plt.ylabel("z-score")
    i = int((mean / std).max(0).indices)
    plt.text(effect_size[i], z_score[i] * 1.1, dataset["mutations"][i], ha="center", fontsize=8)
    plt.title(f"Volcano plot of {len(mean)} mutations")
    plt.savefig("./Results/sars_cov2/volcano_{}".format(gname))


def init_loc_fn(site):
    shape = site["fn"].shape()
    if site["name"].endswith("_scale"):
        return torch.ones(shape)
    if site["name"] == "coef":
        return torch.randn(shape).sub_(0.5).mul(0.01)
    if site["name"] == "rate":
        return torch.zeros(shape)
    if site["name"] == "init":
        return dataset["counts"].mean(0).add(0.01).log()
    raise NotImplementedError(f"TODO initialize latent variable {site['name']}")

def local_guide(dataset):
    # Create learnable parameters.
    T, P, S = dataset["counts"].shape
    r_loc = pyro.param("rate_decentered_loc", lambda: torch.zeros(P, S))
    i_loc = pyro.param("init_decentered_loc", lambda: torch.zeros(P, S))
    skew = pyro.param("skew", lambda: torch.zeros(P, S))  # allows correlation
    r_scale = pyro.param("rate_decentered_scale", lambda: torch.ones(P, S),
                          constraint=constraints.softplus_positive)
    i_scale = pyro.param("init_decentered_scale", lambda: torch.ones(P, S),
                          constraint=constraints.softplus_positive)

    # Sample local variables inside plates.
    # Note plates are already created by the main guide, so we'll
    # use the existing plates rather than calling pyro.plate(...).
    with guide.plates["place"], guide.plates["strain"]:
        samples = {}
        samples["rate_decentered"] = pyro.sample(
            "rate_decentered", dist.Normal(r_loc, r_scale)
        )
        i_loc = i_loc + skew * samples["rate_decentered"]
        samples["init_decentered"] = pyro.sample(
            "init_decentered", dist.Normal(i_loc, i_scale)
        )
    return samples

if __name__ == "__main__":
    args = parse()
    dataset = load_nextstrain_counts()
    reparam_model = poutine.reparam(
    model, {"rate": LocScaleReparam(), "init": LocScaleReparam()}
)
    guide = AutoGuideList(reparam_model)
    local_vars = ["rate_decentered", "init_decentered"]
    guide.add(
        guide_list[args.method](
            poutine.block(reparam_model, hide=local_vars),
            init_loc_fn=init_loc_fn
        )
    )
    guide.add(local_guide)
    fit_svi(reparam_model, guide, args.method, lr=args.lr)
    pprint(evaluate(reparam_model, guide))
    plot_volcano(guide, args.method)