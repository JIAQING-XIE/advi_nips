import pandas as pd
import numpy as np
import torch 
import pyro
import pyro.distributions as dist

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

class BayesModel():
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name
    
    def model(self):
        if self.name == "income":
            is_cont_africa, ruggedness, log_gdp = self.dataset[0], self.dataset[1], self.dataset[2]
            a1 = pyro.sample("a1", dist.Normal(0., 10.))
            a2 = pyro.sample("a2", dist.Normal(0., 1.))
            a3 = pyro.sample("a3", dist.Normal(0., 1.))
            a4 = pyro.sample("a4", dist.Normal(0., 1.))
            sc = pyro.sample("sc", dist.Uniform(0., 10.))
            val = a1 + a2 * is_cont_africa + a3 * ruggedness + a4 * is_cont_africa * ruggedness

            with pyro.plate("data", len(ruggedness)):
                return pyro.sample("obs", dist.Normal(val, sc), obs=log_gdp)
 
