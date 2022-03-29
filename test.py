
import pyro
import torch
import os
import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from Datasets.BayesReg import NationalIncome
from Datasets.VAE import mnist, train_module
from Datasets.utils.mnist_cached import MNISTCached as MNIST
from Datasets.utils.mnist_cached import setup_data_loaders
from Datasets.utils.vae_plots import mnist_test_tsne, plot_llk, plot_vae_samples

import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.infer.autoguide.guides import AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from pyro.infer.autoguide.structured import AutoStructured
from Guides.CirculantMultivariate import CirculantMultivariateNorm
# Datasets
from Datasets.BayesReg import BayesModel, SARS_COV_2
from Datasets.VAE import VAE, mnist, train_module



from collections import defaultdict
from pprint import pprint
import functools
import math
import os
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import (
    AutoDelta,
    AutoNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoGuideList,
    init_to_feasible,
)
from pyro.infer.reparam import AutoReparam, LocScaleReparam
from pyro.nn.module import PyroParam
from pyro.optim import ClippedAdam
from pyro.ops.special import sparse_multinomial_likelihood
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print("Using GPU")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("Using CPU")

pyro.distributions.enable_validation(False)
smoke_test = ('CI' in os.environ)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

assert pyro.__version__.startswith('1.8.1')

pyro.enable_validation(True)
pyro.set_rng_seed(1)
# logging.basicConfig(format='%(message)s', level=logging.INFO)
plt.style.use('default')



def parse():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type = str, help = 'dataset')
    parser.add_argument('--method', type = str, help = 'autoguide methods')
    parser.add_argument('--epochs', type = int, help = 'number of epochs')
    parser.add_argument('--lr_rate', default= 0.02, type=float, help = "learning rate")
    parser.add_argument('--optimizer', default = "Adam", type = str, help= "optimizer")
    parser.add_argument('--step_lr', default = False, type = bool, help = "if use step scheduler")
    parser.add_argument('--expo_lr', default = False, type = bool, help = "if use exponential optimizer")
    parser.add_argument('--gamma', default= 0.1, type=float, help = "gamma")
    return parser.parse_args()

common_set = {
    "circulant": CirculantMultivariateNorm,
    "autonormal": AutoDiagonalNormal,
    "structured": AutoStructured
}


# this is the svi training for sars-cov-2
def fit_svi(model, guide, dataset, lr=0.01, num_steps=1001, log_every=100, plot=True):
    pyro.clear_param_store()
    pyro.set_rng_seed(20220322)
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
        print(loss)
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
        plt.show()


if __name__ == "__main__":
    args = parse()
    # set optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD
    # set scheduler
    if args.expo_lr:
        scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': {'lr': args.lr_rate}, 'gamma': args.gamma})
    elif args.step_lr:
        scheduler = pyro.optim.StepLR({'optimizer': optimizer, 'optim_args': {'lr': args.lr_rate}, 'step_size': 40, 'gamma': args.gamma})
    # srt ELBO
    ELBO = pyro.infer.Trace_ELBO()

    # set the dataset: guide and model
    if args.dataset == "income":
        NI = NationalIncome()
        dataset = NI.read_data()
        dataset = NI.split_data(data = dataset)
        model = BayesModel(dataset = dataset, name = args.dataset).model
        guide = common_set[args.method](model)
    #    adam = pyro.optim.Adam({"lr": 0.02})
    #elbo = pyro.infer.Trace_ELBO()
#svi = pyro.infer.SVI(model, guide, adam, elbo)
    #print(dataset)
        if args.step_lr or args.expo_lr:
            svi = pyro.infer.SVI(model, guide, scheduler, ELBO)
        else:
            svi = pyro.infer.SVI(model, guide, optimizer, ELBO) 
        losses = []
        best_loss = 10000001
        for step in range(1000 if not smoke_test else 2):  # Consider running for more steps.
            loss = svi.step()
            losses.append(loss)
            print(loss)
            if loss < best_loss:
                guide.print_best_cov()
                torch.save({"model" : model, "guide" : guide}, "./best_models/mymodel.pt")
                pyro.get_param_store().save("./best_models/saved_params.save")
                best_loss = loss
            if step % 100 == 0:
                logging.info("Elbo loss: {}".format(loss))
                
        plt.figure(figsize=(5, 2))
        plt.plot(losses)
        plt.xlabel("SVI step")
        plt.ylabel("ELBO loss")
        plt.show()

        print("Best parameter ")
        pyro.clear_param_store()
        """
        saved_model_dict = torch.load("mymodel.pt")
        model.load_state_dict(saved_model_dict['model'])
        guide = saved_model_dict['guide']
        """
        pyro.get_param_store().load('./best_models/saved_params.save')
        #print(pyro.get_param_store())
        for key, value in pyro.get_param_store().items():    
            print(f"{key}:\n{value}\n")
        
    elif args.dataset == "covid":
        sars = SARS_COV_2()
        sars.read_data()
        sars.summarize(sars.dataset)
        def init_loc_fn(site, dataset = sars.dataset):
            shape = site["fn"].shape()
            if site["name"] == "coef":
                return torch.randn(shape).sub_(0.5).mul(0.01)
            if site["name"] == "init":
                # Heuristically initialize based on data.
                return dataset["counts"].mean(0).add(0.01).log()
            return init_to_feasible(site)  # fallback

        BM = BayesModel(sars.dataset, "covid")
        reparam_model = poutine.reparam(
            BM.model, {"rate": LocScaleReparam(), "init": LocScaleReparam()}
        )
        guide = AutoGuideList(reparam_model)
        mvn_vars = ["coef", "rate_scale", "coef_scale"]
        guide.add(
            CirculantMultivariateNorm(
                poutine.block(reparam_model, expose=mvn_vars),
                init_scale=0.1, residual=0.1
            )
        )
        guide.add(
            AutoNormal(
                poutine.block(reparam_model, hide=mvn_vars),
                init_loc_fn=init_loc_fn,
                init_scale=0.01,
            )
        )

        #guide = AutoMultivariateNormal(BM.model, init_scale=0.01)

        fit_svi(BM.model, guide, sars.dataset, lr = 0.1)

    elif args.dataset == "mnist":
        pyro.clear_param_store()
        TEST_FREQUENCY = 5
        #train_loader, test_loader = MNIST.split_data(batch_size=256, use_cuda=False)
        train_loader, test_loader = setup_data_loaders(
        MNIST, use_cuda=False, batch_size=256
    )
        vae = VAE(use_cuda=False)
        #guide = AutoDiagonalNormal(vae)
        svi = SVI(vae.model, vae.guide, scheduler, loss = Trace_ELBO())
        
        tm = train_module(svi)

        pyro.clear_param_store()
        print("yes")

        train_elbo = []
        test_elbo = []
        # training loop
        for epoch in range(200):
            epoch_loss = 0.0
            # do a training epoch over each mini-batch x returned
            # by the data loader
            for x, _ in train_loader:
                # if on GPU put mini-batch into CUDA memory
                # do ELBO gradient and accumulate loss
                epoch_loss += svi.step(x)

            """
            total_epoch_loss_train = tm.train(train_loader)
            train_elbo.append(-total_epoch_loss_train)
            print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

            if epoch % TEST_FREQUENCY == 0:
                # report test diagnostics
                total_epoch_loss_test = tm.evaluate(test_loader)
                test_elbo.append(-total_epoch_loss_test)
                print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
            """

