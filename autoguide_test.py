from random import random
import pyro
import pandas as pd
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import os

import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.infer.autoguide.guides import AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from Models.PolyDiagonalNormal import PolyDiagNorm
from Models.OrthoMultivariateNormal import OrthoMultiNorm
from Models.LowRankNormal import LowRankNormal
from pyro.infer.autoguide.structured import AutoStructured

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.0')

pyro.enable_validation(True)
pyro.set_rng_seed(1)
# logging.basicConfig(format='%(message)s', level=logging.INFO)

plt.style.use('default')

# import pdb;pdb.set_trace()
# linear regression model:
### dataset from "Geography and national income"
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
# log transform
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
# transform to torch tensor
train = torch.tensor(df.values, dtype=torch.float)
is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]


def model(is_cont_africa, ruggedness, log_gdp=None):
    a1 = pyro.sample("a1", dist.Normal(0., 10.))
    a2 = pyro.sample("a2", dist.Normal(0., 1.))
    a3 = pyro.sample("a3", dist.Normal(0., 1.))
    a4 = pyro.sample("a4", dist.Normal(0., 1.))
    sc = pyro.sample("sc", dist.Uniform(0., 10.))
    val = a1 + a2 * is_cont_africa + a3 * ruggedness + a4 * is_cont_africa * ruggedness

    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(val, sc), obs=log_gdp)

guide = LowRankNormal(model, random = True)
#guide = AutoStructured(model,
#            conditionals={"a1": "normal", "a2": "normal", "a3": "normal", "a4": "normal", "sc": "normal"},
#            dependencies={"a1": {"a2": "linear", "a3":"linear"}, "a4": {"a3":"linear"}})
#guide = OrthoMultiNorm(model)
#guide = PolyDiagNorm(model, epoch = 1000, order = 1)
#guide = AutoMultivariateNormal(model)
#guide = AutoDiagonalNormal(model)
#guide = AutoLowRankMultivariateNormal(model, rank = 5)
## train
adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

pyro.clear_param_store()

losses = []
for step in range(2000 if not smoke_test else 2):  # Consider running for more steps.
    loss = svi.step(is_cont_africa, ruggedness, log_gdp)
    losses.append(loss)
    print(loss)

    if step % 100 == 0:
        logging.info("Elbo loss: {}".format(loss))

plt.figure(figsize=(5, 2))
plt.plot(losses)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss")
plt.show()

for key, value in pyro.get_param_store().items():    
    print(f"{key}:\n{value}\n")