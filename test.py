
import pyro
import torch
import os
import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from Datasets.BayesReg import NationalIncome
from Datasets.VAE import mnist


import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.infer.autoguide.guides import AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from pyro.infer.autoguide.structured import AutoStructured
from Guides.CirculantMultivariate import CirculantMultivariateNorm
from Datasets.BayesReg import BayesianRegression, BayesModel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.0')

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
    "circulant": CirculantMultivariateNorm
}

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
