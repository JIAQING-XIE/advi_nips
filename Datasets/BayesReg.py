import pandas as pd
import numpy as np
import torch 

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
            return is_cont_africa, ruggedness, log_gdp 
        else:
            raise NotImplementedError("Do not support the variables other than is_cont_africa, ruggedness, and log_gdp")
