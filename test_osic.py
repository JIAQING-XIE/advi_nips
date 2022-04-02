import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyro.distributions as dist
import pyro
import torch
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
from pyro.infer import  Predictive, SVI, Trace_ELBO
from Guides.PolyDiagonalNormal import PolyDiagNorm
from Guides.ToeplitzMultivariate import ToeplitzMultivariateNorm
from pyro.infer import MCMC
from pyro.optim import ClippedAdam
import arviz as az

train = pd.read_csv(
    "https://gist.githubusercontent.com/ucals/"
    "2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/"
    "43034c39052dcf97d4b894d2ec1bc3f90f3623d9/"
    "osic_pulmonary_fibrosis.csv"
)

def chart(patient_id, ax):
    data = train[train["Patient"] == patient_id]
    x = data["Weeks"]
    y = data["FVC"]
    ax.set_title(patient_id)
    ax = sns.regplot(x, y, ax=ax, ci=None, line_kws={"color": "red"})


f, axes = plt.subplots(1, 3, figsize=(15, 5))
chart("ID00007637202177411956430", axes[0])
chart("ID00009637202177434476278", axes[1])
chart("ID00010637202177584971671", axes[2])
#plt.show()

def model(PatientID, Weeks, FVC_obs=None):
    μ_α = pyro.sample("μ_α", dist.Normal(0.0, 100.0))
    σ_α = pyro.sample("σ_α", dist.HalfNormal(100.0))
    μ_β = pyro.sample("μ_β", dist.Normal(0.0, 100.0))
    σ_β = pyro.sample("σ_β", dist.HalfNormal(100.0))

    unique_patient_IDs = np.unique(PatientID)
    n_patients = len(unique_patient_IDs)

    with pyro.plate("plate_i", n_patients):
        α = pyro.sample("α", dist.Normal(μ_α, σ_α))
        β = pyro.sample("β", dist.Normal(μ_β, σ_β))

    σ = pyro.sample("σ", dist.HalfNormal(100.0))

    FVC_est = α[PatientID] + β[PatientID] * Weeks

    with pyro.plate("data", len(PatientID)):
        pyro.sample("obs", dist.Normal(FVC_est, σ), obs=FVC_obs)



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train["PatientID"] = le.fit_transform(train["Patient"].values)

FVC_obs = torch.Tensor(train["FVC"].values)
Weeks = torch.Tensor(train["Weeks"].values)
PatientID = train["PatientID"].values
from pyro.infer.autoguide import AutoStructured
guide = AutoStructured(model)
optim = ClippedAdam({"lr": 0.01, "betas": (0.9, 0.99), "lrd": 1.0})
svi = SVI(model, guide, optim, Trace_ELBO())

best_loss = 10001
for i in range(1200):
    loss = svi.step(PatientID, Weeks, FVC_obs)
    if loss < best_loss:
        best_loss = loss
    #    pyro.get_param_store().save("./best_params/gmm/saved_params_{}_{}.save".format(args.method, args.K))
    #losses.append(loss)
    print('.' if i % 100 else '\n', end='')
    if i % 100 == 0:
        print(loss) 

print("Best loss: {}".format)
predictive = Predictive(model, guide=guide, num_samples=500)
preds = predictive(PatientID, Weeks, FVC_obs)
sanitized_preds = {k: v.unsqueeze(0).detach().numpy() for k, v in preds.items() if k != 'obs'}
pyro_data = az.convert_to_inference_data(sanitized_preds)
axes = az.plot_trace(pyro_data, compact=True)
fig = axes.ravel()[0].figure
fig.savefig("./Results/osic/trace.png")