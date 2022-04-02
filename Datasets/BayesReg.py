import pandas as pd
import numpy as np
import torch 
import pyro
import pyro.distributions as dist
from pyro.contrib.examples.nextstrain import load_nextstrain_counts
from pyro.ops.special import sparse_multinomial_likelihood
import matplotlib.pyplot as plt


class BayesianRegression():
    def __init__(self, url):
        self.url = url

    def read_data(self):
        raise NotImplementedError("Not being implemented")
    
    def split_data(self):
        raise NotImplementedError("Not being implemented")
        
class NationalIncome(BayesianRegression):
    def __init__(self):
        self.url = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"

    def read_data(self):
        dataset = pd.read_csv(self.url,  encoding="ISO-8859-1")
        return dataset
    
    def split_data(self, data = None, default = True):
        if default:
            """ in default settings, only three variables are chosen"""
            df = data[["cont_africa", "rugged", "rgdppc_2000"]]
            # log transform
            df = df[np.isfinite(df.rgdppc_2000)]
            df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
            # transform to torch tensor
            train = torch.tensor(df.values, dtype=torch.float)
            is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]
            return [is_cont_africa, ruggedness, log_gdp] 
        else:
            raise NotImplementedError("Do not support the variables other than is_cont_africa, ruggedness, and log_gdp")

class SARS_COV_2(BayesianRegression):
    def __init__(self):
        self.url = None

    def read_data(self):
        self.dataset = load_nextstrain_counts()

    def summarize(self, x, name=""):
        if isinstance(x, dict):
            for k, v in sorted(x.items()):
                self.summarize(v, name + "." + k if name else k)
        elif isinstance(x, torch.Tensor):
            print(f"{name}: {type(x).__name__} of shape {tuple(x.shape)} on {x.device}")
        elif isinstance(x, list):
            print(f"{name}: {type(x).__name__} of length {len(x)}")
        else:
            print(f"{name}: {type(x).__name__}")

class BayesModel():
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name
    
    def model(self, is_cont_africa, ruggedness, log_gdp = None):
        if self.name == "income":
            a1 = pyro.sample("a1", dist.Normal(0., 10.))
            a2 = pyro.sample("a2", dist.Normal(0., 1.))
            a3 = pyro.sample("a3", dist.Normal(0., 1.))
            a4 = pyro.sample("a4", dist.Normal(0., 1.))
            sc = pyro.sample("sc", dist.Uniform(0., 10.))
            val = a1 + a2 * is_cont_africa + a3 * ruggedness + a4 * is_cont_africa * ruggedness

            with pyro.plate("data", len(ruggedness)):
                return pyro.sample("obs", dist.Normal(val, sc), obs=log_gdp)
    