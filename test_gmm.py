from cProfile import label
import os
import torch
import random
import pyro
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot
import pyro.distributions as dist
import torch.distributions as tdist
from pyro import poutine
from scipy import stats
from collections import defaultdict
from pyro.infer.autoguide import AutoDiagonalNormal, AutoStructured
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from Guides.guide_list import guide_list

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.1')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def parse():
    parser = ArgumentParser()
    parser.add_argument('--method', type = str, help = 'autoguide methods')
    parser.add_argument('--epochs', default = 200, type = int, help = 'number of epochs')
    parser.add_argument('--lr', default= 0.1, type=float, help = "learning rate")
    parser.add_argument('--step_lr', default = False, type = bool, help = "if use step scheduler")
    parser.add_argument('--expo_lr', default = False, type = bool, help = "if use exponential optimizer")
    parser.add_argument('--gamma', default= 0.1, type=float, help = "gamma")
    parser.add_argument('--K', default = 2, type = int, help = "number of centers")
    parser.add_argument('--order', default=1, type = int, help="order for PolyDiag")
    return parser.parse_args()

# define the length of data according to the number of centers
len_data = {
    2: 20,
    3: 60,
    5: 100
}

def generate_dataset(K, scale = 1.5):
    """ generate data according to the number of centers, return data, centers and scale"""
    random.seed(2022)
    centers = [random.randint(0, len_data[K]) for i in range(K)]
    print("Centers are: {}".format(centers))
    num_of_data = int (len_data[K] / K)
    d = torch.tensor([])
    torch.manual_seed(2022) 
    for center in centers:   
        data = tdist.Normal(torch.tensor([float(center)]), torch.tensor([scale])).sample((num_of_data,))
        d = torch.cat((d, data), 0)
    d = d.reshape((K * num_of_data,))
    print(d)
    return d, centers, scale

    
if __name__ == "__main__":
    args = parse()
    data, centers, scale = generate_dataset(args.K)

    @config_enumerate
    def model(data):
        # Global variables.
        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(args.K)))
        scale = pyro.sample('scale', dist.LogNormal(0., 2.))
        with pyro.plate('components', args.K):
            locs = pyro.sample('locs', dist.Normal(50., 10.))

        with pyro.plate('data', len(data)):
            # Local variables.
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)

    optim = pyro.optim.Adam({'lr': args.lr, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)

    def init_loc_fn(site):
        if site["name"] == "weights":
            # Initialize weights to uniform.
            return torch.ones(args.K) / args.K
        if site["name"] == "scale":
            return (data.var() / 2).sqrt()
        if site["name"] == "locs":
            return data[torch.multinomial(torch.ones(len(data)) / len(data), args.K)]
        raise ValueError(site["name"])

    def initialize(seed):
        global global_guide, svi
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        guide = guide_list[args.method]
        if args.method == "polydiag":
            global_guide = guide(poutine.block(model, expose=['weights', 'locs', 'scale']),
                                init_loc_fn=init_loc_fn, order = args.order)
        elif args.method == "structured":
            global_guide = guide(poutine.block(model, expose=['weights', 'locs', 'scale']),
                                init_loc_fn=init_loc_fn, dependencies={"weights": {"locs": "linear"},},
                                conditionals = {"weights": "normal", "scale": "delta", "locs": "normal"})
        else:
            global_guide = guide(poutine.block(model, expose=['weights', 'locs', 'scale']),
                                init_loc_fn=init_loc_fn)
        svi = SVI(model, global_guide, optim, loss=elbo)
        return svi.loss(model, global_guide, data)

    # Choose the best among 100 random initializations.
    loss, seed = min((initialize(seed), seed) for seed in range(100))
    initialize(seed)
    print('seed = {}, initial_loss = {}'.format(seed, loss))

    # Register hooks to monitor gradient norms.
    gradient_norms = defaultdict(list)
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    losses = []
    best_loss = 1000001
    for i in range(args.epochs if not smoke_test else 2):
        loss = svi.step(data)
        if loss < best_loss:
            best_loss = loss
        losses.append(loss)
        print('.' if i % 100 else '\n', end='')
    
    print("Best loss: {}".format(best_loss))
    pyplot.figure(figsize=(16,8), dpi=100).set_facecolor('white')
    pyplot.plot(losses)
    pyplot.xlabel('iters')
    pyplot.ylabel('loss')
    pyplot.yscale('log')
    pyplot.title('Convergence of SVI on: {}'.format(args.method))
    pyplot.savefig("./Results/gmm/svi_loss_{}_{}.png".format(args.method, args.K))



    map_estimates = global_guide(data)
    weights = map_estimates['weights']
    locs = map_estimates['locs']
    scale = map_estimates['scale']
    print('weights = {}'.format(weights.data.numpy()))
    print('locs = {}'.format(locs.data.numpy()))
    print('scale = {}'.format(scale.data.numpy()))

    if args.K <= 3:
        X = np.arange(0,max(data) + 2,0.1)
        if args.method == "polydiag":
            Y1 = weights[0][0].item() * stats.norm.pdf((X - locs[0].item()) / scale[0].item())
            Y2 = weights[0][1].item() * stats.norm.pdf((X - locs[1].item()) / scale[0].item())
        else:
            Y1 = weights[0].item() * stats.norm.pdf((X - locs[0].item()) / scale.item())
            Y2 = weights[1].item() * stats.norm.pdf((X - locs[1].item()) / scale.item())
        pyplot.figure(figsize=(16, 8), dpi=100).set_facecolor('white')
        pyplot.plot(X, Y1, 'r-', label = "center1: {}".format(centers[0]))
        pyplot.plot(X, Y2, 'b-', label = "center2: {}".format(centers[1]))
        if args.K == 3:
            if args.method == "polydiag":
                Y3 = weights[0][2].item() * stats.norm.pdf((X - locs[2].item()) / scale[0].item())
            else:
                Y3 = weights[2].item() * stats.norm.pdf((X - locs[2].item()) / scale.item())
            pyplot.plot(X, Y3, 'g-', label = "center3: {}".format(centers[2]))
        #pyplot.plot(X, Y1 + Y2, 'k--')
        pyplot.plot(data.data.numpy(), np.zeros(len(data)), 'k*')
        if args.K == 2:
            pyplot.title('Density of two-component mixture model : {}'.format(args.method))
        elif args.K == 3:
            pyplot.title('Density of three-component mixture model : {}'.format(args.method))
        pyplot.ylabel('probability density')
        pyplot.legend()
        pyplot.savefig("./Results/gmm/density_{}_{}.png".format(args.method, args.K))