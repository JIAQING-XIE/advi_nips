import os
import math
from xmlrpc.client import boolean
import torch
import pyro
import pyro.distributions as dist
from matplotlib import pyplot
from argparse import ArgumentParser
from torch.distributions import constraints
from pyro import poutine
from pyro.contrib.examples.finance import load_snp500
from pyro.infer import EnergyDistance, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.reparam import DiscreteCosineReparam, StableReparam
from pyro.optim import ClippedAdam
from pyro.ops.tensor_utils import convolve
from Guides.guide_list import guide_list

assert pyro.__version__.startswith('1.8.1')
smoke_test = ('CI' in os.environ)

def parse():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default = "income", type = str,help = 'dataset')
    parser.add_argument('--method', type = str, help = 'autoguide methods')
    parser.add_argument('--epochs', default = 501, type = int, help = 'number of epochs')
    parser.add_argument('--optimizer', default="Adam", type= str, help="customize your optimizer")
    parser.add_argument('--lr_rate', default= 0.001, type=float, help = "learning rate")
    parser.add_argument('--length', default=100, type = int, help = "length of recorded days")
    parser.add_argument('--plot_data', default= True, type=boolean, help = "plot data")
    parser.add_argument('--plot_ratio', default= True, type=boolean, help = "plot ratio between two sequent days")
    parser.add_argument('--plot_dist', default= True, type=boolean, help = "plot emp dist")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    # dataset
    df = load_snp500()
    dates = df.Date.to_numpy()[-args.length:]
    x = torch.tensor(df["Close"]).float()[-args.length:]

    if args.plot_data:
        pyplot.figure(figsize=(9, 3))
        pyplot.plot(x)
        pyplot.yscale('log')
        pyplot.ylabel("index")
        pyplot.xlabel("trading day")
        pyplot.title("S&P 500 from {} to {}".format(dates[0], dates[-1]))
        pyplot.savefig("./Results/levy/data_{}.png".format(args.length), dpi = 100) 

    if args.plot_ratio:
        pyplot.figure(figsize=(9, 3))
        r = (x[1:] / x[:-1]).log()
        pyplot.plot(r, "k", lw=0.1)
        pyplot.title("daily log returns")
        pyplot.xlabel("trading day")
        pyplot.savefig("./Results/levy/data_ratio_{}.png".format(args.length), dpi = 100)

    if args.plot_dist:
        pyplot.figure(figsize=(9, 3))
        pyplot.hist(r.numpy(), bins=200)
        pyplot.yscale('log')
        pyplot.ylabel("count")
        pyplot.xlabel("daily log returns")
        pyplot.title("Empirical distribution.  mean={:0.3g}, std={:0.3g}".format(r.mean(), r.std()))
        pyplot.savefig("./Results/levy/emp_dist_{}.png".format(args.length), dpi = 100)

    def model(data):
        h_0 = pyro.sample("h_0", dist.Normal(0, 1)).unsqueeze(-1)
        sigma = pyro.sample("sigma", dist.LogNormal(0, 1)).unsqueeze(-1)
        v = pyro.sample("v", dist.Normal(0, 1).expand(data.shape).to_event(1))
        log_h = pyro.deterministic("log_h", h_0 + sigma * v.cumsum(dim=-1))
        sqrt_h = log_h.mul(0.5).exp().clamp(min=1e-8, max=1e8)

        # Observed log returns, assumed to be a Stable distribution scaled by sqrt(h).
        r_loc = pyro.sample("r_loc", dist.Normal(0, 1e-2)).unsqueeze(-1)
        r_skew = pyro.sample("r_skew", dist.Uniform(-1, 1)).unsqueeze(-1)
        r_stability = pyro.sample("r_stability", dist.Uniform(0, 2)).unsqueeze(-1)
        pyro.sample("r", dist.Stable(r_stability, r_skew, sqrt_h, r_loc * sqrt_h).to_event(1),
                    obs=data)

    reparam_model = poutine.reparam(model, {"v": DiscreteCosineReparam(),
                                            "r": StableReparam()})

    pyro.clear_param_store()
    pyro.set_rng_seed(1234567890)
    num_steps = 1 if smoke_test else args.epochs
    optim = ClippedAdam({"lr": args.lr_rate, "betas": (0.9, 0.99), "lrd": 0.8})
    if args.method == "symmetric":
        guide = guide_list[args.method](reparam_model, residual = 0.001)
    else:
        guide = guide_list[args.method](reparam_model)
    svi = SVI(reparam_model, guide, optim, Trace_ELBO())
    losses = []
    best_loss = 1000001
    for step in range(num_steps):
        loss = svi.step(r) / len(r)
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            pyro.get_param_store().save("./best_params/levy/best_model_{}.save".format(args.method))
        if step % 50 == 0:
            #median = guide.median()
            print("step {} loss = {:0.6g}".format(step, loss))
    print("Best loss: {}".format(best_loss))
    print("-" * 20)
    for name, (lb, ub) in sorted(guide.quantiles([0.325, 0.675]).items()):
        if lb.numel() == 1:
            lb = lb.squeeze().item()
            ub = ub.squeeze().item()
            print("{} = {:0.4g} ± {:0.4g}".format(name, (lb + ub) / 2, (ub - lb) / 2))

    pyplot.figure(figsize=(9, 3))
    pyplot.plot(losses)
    pyplot.ylabel("loss")
    pyplot.xlabel("SVI step")
    pyplot.xlim(0, len(losses))
    pyplot.ylim(min(losses), 20)
    pyplot.savefig("./Results/levy/loss_{}_{}.png".format(args.method, args.length), dpi = 100)

    pyro.clear_param_store()
    pyro.get_param_store().load("./best_params/levy/best_model_{}.save".format(args.method))
    fig, axes = pyplot.subplots(2, figsize=(9, 5), sharex=True)
    pyplot.subplots_adjust(hspace=0)
    axes[1].plot(r, "k", lw=0.2)
    axes[1].set_ylabel("log returns")
    axes[1].set_xlim(0, len(r))

    # We will pull out median log returns using the autoguide's .median() and poutines.
    with torch.no_grad():
        pred = Predictive(reparam_model, guide=guide, num_samples=20, parallel=True)(r)
    log_h = pred["log_h"]
    axes[0].plot(log_h.median(0).values, lw=1)
    axes[0].fill_between(torch.arange(len(log_h[0])),
                        log_h.kthvalue(2, dim=0).values,
                        log_h.kthvalue(18, dim=0).values,
                        color='red', alpha=0.5)
    axes[0].set_ylabel("log volatility")

    stability = pred["r_stability"].median(0).values.item()
    axes[0].set_title("Estimated index of stability = {:0.4g}".format(stability))
    axes[1].set_xlabel("trading day")
    pyplot.savefig("./Results/levy/est_log_return_{}.png".format(args.method))