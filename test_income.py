import pyro
import torch
import os
import logging
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from Datasets.BayesReg import NationalIncome

from Guides.guide_list import guide_list
from Datasets.BayesReg import BayesModel
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
logging.basicConfig(format='%(message)s', level=logging.INFO)
plt.style.use('default')

def parse():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default = "income", type = str,help = 'dataset')
    parser.add_argument('--method', type = str, help = 'autoguide methods')
    parser.add_argument('--epochs', default = 2000, type = int, help = 'number of epochs')
    parser.add_argument('--optimizer', default="Adam", type= str, help="customize your optimizer")
    parser.add_argument('--lr_rate', default= 0.1, type=float, help = "learning rate")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    # set optimizer
    if args.optimizer == "Adam":
        optimizer = pyro.optim.Adam({"lr": args.lr_rate})
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD({"lr": args.lr_rate})
    # srt ELBO
    ELBO = pyro.infer.Trace_ELBO()

    # set the dataset: guide and model
    if args.dataset == "income":
        NI = NationalIncome()
        dataset = NI.read_data()
        dataset = NI.split_data(data = dataset)
        model = BayesModel(dataset = dataset, name = args.dataset).model
        guide = guide_list[args.method](model)

        svi = pyro.infer.SVI(model, guide, optimizer, ELBO) 
        losses = []
        best_loss = 10000001
        for step in range(args.epochs if not smoke_test else 2):  # Consider running for more steps.
            loss = svi.step(dataset[0], dataset[1], dataset[2])
            losses.append(loss)
            if loss < best_loss:
                pyro.get_param_store().save("./best_params/income/income_{}".format(args.method))
                best_loss = loss

        

        print("best loss: {:.4f}".format(best_loss))
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.xlabel("SVI step")
        plt.ylabel("ELBO loss")
        plt.savefig("./Results/income/loss_{}_{}.png".format(args.method, args.lr_rate), dpi =100)

        print("Best parameter ")
        pyro.clear_param_store()

        pyro.get_param_store().load("./best_params/income/income_{}".format(args.method))
        #print(pyro.get_param_store())
        for key, value in pyro.get_param_store().items():    
            print(f"{key}:\n{value}\n")

        if args.method == "lowrank" or args.method == "toeplitz":
            with pyro.plate("samples", 800, dim=-1):
                samples = guide(dataset[0], dataset[1])

                gamma_within_africa = samples["a3"] + samples["a4"]
                gamma_outside_africa = samples["a3"]

            fig = plt.figure(figsize=(10, 6))
            sns.histplot(gamma_within_africa.detach().cpu().numpy(), kde=True, stat="density", label="African nations")
            sns.histplot(gamma_outside_africa.detach().cpu().numpy(), kde=True, stat="density", label="Non-African nations", color="orange")
            fig.suptitle("Density of Slope : log(GDP) vs. Terrain Ruggedness");
            plt.xlabel("Slope of regression line")
            plt.legend()
            plt.savefig("./Results/income/income_dist_{}".format(args.method), dpi = 100)

            predictive = pyro.infer.Predictive(model, guide=guide, num_samples=800)
            svi_samples = predictive(dataset[0], dataset[1], log_gdp=None)
            svi_gdp = svi_samples["obs"]

            predictions = pd.DataFrame({
                "cont_africa": dataset[0],
                "rugged": dataset[1],
                "y_mean": svi_gdp.mean(0).detach().cpu().numpy(),
                "y_perc_5": svi_gdp.kthvalue(int(len(svi_gdp) * 0.05), dim=0)[0].detach().cpu().numpy(),
                "y_perc_95": svi_gdp.kthvalue(int(len(svi_gdp) * 0.95), dim=0)[0].detach().cpu().numpy(),
                "true_gdp": dataset[2],
            })
            african_nations = predictions[predictions["cont_africa"] == 1].sort_values(by=["rugged"])
            non_african_nations = predictions[predictions["cont_africa"] == 0].sort_values(by=["rugged"])

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
            fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)

            ax[0].plot(non_african_nations["rugged"], non_african_nations["y_mean"])
            ax[0].fill_between(non_african_nations["rugged"], non_african_nations["y_perc_5"], non_african_nations["y_perc_95"], alpha=0.5)
            ax[0].plot(non_african_nations["rugged"], non_african_nations["true_gdp"], "o")
            ax[0].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="Non African Nations")

            ax[1].plot(african_nations["rugged"], african_nations["y_mean"])
            ax[1].fill_between(african_nations["rugged"], african_nations["y_perc_5"], african_nations["y_perc_95"], alpha=0.5)
            ax[1].plot(african_nations["rugged"], african_nations["true_gdp"], "o")
            ax[1].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations")
            plt.savefig("./Results/income/income_post_{}".format(args.method), dpi = 100)