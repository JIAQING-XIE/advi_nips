import pyro
import torch
import os
import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from Datasets.BayesReg import NationalIncome

from Guides.guide_list import guide_list

# Datasets
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
            loss = svi.step()
            losses.append(loss)
            if loss < best_loss:
                pyro.get_param_store().save("./best_params/income/income_{}".format(args.method))
                best_loss = loss
            #if step % 100 == 0:
            #    logging.info("Elbo loss: {}".format(loss))
        

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
