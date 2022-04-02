from xmlrpc.client import boolean
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyro.distributions as dist
import pyro
import torch
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
from pyro.infer import  Predictive, SVI, Trace_ELBO
from pyro.infer import MCMC
from pyro.optim import ClippedAdam
import arviz as az
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder
from Guides.guide_list import guide_list


def parse():
    parser = ArgumentParser()
    parser.add_argument('--method', type = str, help = 'autoguide methods')
    parser.add_argument('--epochs', default = 2001, type = int, help = 'number of epochs')
    parser.add_argument('--lr_rate', default= 0.11, type=float, help = "learning rate")
    parser.add_argument('--order', default=1, type = int, help="order for PolyDiag")
    parser.add_argument('--vis_pred', default=True, type = bool, help= "if visualize prediction")
    parser.add_argument('--vis_data', default=False, type = bool, help= "if visualize data")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()

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
    chart("ID00099637202206203080121", axes[1])
    chart("ID00010637202177584971671", axes[2])
    if args.vis_data:
        plt.savefig("./Results/osic/time_FVC.png", dpi = 100)

    def model(PatientID, Weeks, FVC_obs=None):
        μ_α = pyro.sample("μ_α", dist.Normal(3000, 100.0))
        σ_α = pyro.sample("σ_α", dist.HalfNormal(3000.0))
        μ_β = pyro.sample("μ_β", dist.Normal(-5.0, 10.0))
        σ_β = pyro.sample("σ_β", dist.HalfNormal(10.0))

        unique_patient_IDs = np.unique(PatientID)
        n_patients = len(unique_patient_IDs)

        with pyro.plate("plate_i", n_patients):
            α = pyro.sample("α", dist.Normal(μ_α, σ_α))
            β = pyro.sample("β", dist.Normal(μ_β, σ_β))

        σ = pyro.sample("σ", dist.HalfNormal(140.0))

        FVC_est = α[PatientID] + β[PatientID] * Weeks
        #print(FVC_est)
        #print(σ)

        with pyro.plate("data", len(PatientID)):
            pyro.sample("obs", dist.Normal(FVC_est, σ), obs=FVC_obs)

    le = LabelEncoder()
    train["PatientID"] = le.fit_transform(train["Patient"].values)

    FVC_obs = torch.Tensor(train["FVC"].values)
    Weeks = torch.Tensor(train["Weeks"].values)
    PatientID = train["PatientID"].values

    guide = guide_list[args.method](model)
    #optim = pyro.optim.Adam({'lr': args.lr_rate, 'betas': [0.9, 0.99]})
    optim = ClippedAdam({"lr": args.lr_rate, "betas": (0.9, 0.99), "lrd": 1 ** (1/args.epochs)})
    svi = SVI(model, guide, optim, Trace_ELBO())

    best_loss = 1000001
    for i in range(args.epochs):
        loss = svi.step(PatientID, Weeks, FVC_obs)
        if loss < best_loss:
            best_loss = loss
            pyro.get_param_store().save("./best_params/osic/saved_params_{}.save".format(args.method))
        #losses.append(loss)
        print('.' if i % 100 else '\n', end='')
        if i % 100 == 0:
            print(loss) 
    print("Best loss: {}".format(best_loss))
    pyro.clear_param_store()
    pyro.get_param_store().load("./best_params/osic/saved_params_{}.save".format(args.method))
    
    predictive = Predictive(model, guide=guide, num_samples=200)
    preds = predictive(PatientID, Weeks, None)
    sanitized_preds = {k: v.unsqueeze(0).detach().numpy() for k, v in preds.items() if k != 'obs'}
    pyro_data = az.convert_to_inference_data(sanitized_preds)
    axes = az.plot_trace(pyro_data, compact=True)
    fig = axes.ravel()[0].figure
    fig.savefig("./Results/osic/trace_{}.png".format(args.method))
    
    if args.vis_pred:
        pred_template = []
        for i in range(train["Patient"].nunique()):
            df = pd.DataFrame(columns=["PatientID", "Weeks"])
            df["Weeks"] = np.arange(-12, 134)
            df["PatientID"] = i
            pred_template.append(df)
        pred_template = pd.concat(pred_template, ignore_index=True)

        PatientID = pred_template["PatientID"].values
        Weeks = pred_template["Weeks"].values
        predictive = Predictive(model, guide = guide, num_samples = 200,  return_sites=["σ", "obs"])
        samples_predictive = predictive(PatientID, Weeks, None)

        df = pd.DataFrame(columns=["Patient", "Weeks", "FVC_pred", "sigma"])
        df["Patient"] = le.inverse_transform(pred_template["PatientID"])
        df["Weeks"] = pred_template["Weeks"]
        df["FVC_pred"] = samples_predictive["obs"].T.mean(axis=1)
        df["sigma"] = samples_predictive["obs"].T.std(axis=1)
        df["FVC_inf"] = df["FVC_pred"] - df["sigma"]
        df["FVC_sup"] = df["FVC_pred"] + df["sigma"]
        df = pd.merge(
            df, train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
        )
        df = df.rename(columns={"FVC": "FVC_true"})

        def chart(patient_id, ax):
            data = df[df["Patient"] == patient_id]
            x = data["Weeks"]
            ax.set_title(patient_id)
            ax.plot(x, data["FVC_true"], "o")
            ax.plot(x, data["FVC_pred"])
            ax = sns.regplot(x, data["FVC_true"], ax=ax, ci=None, line_kws={"color": "red"})
            ax.fill_between(x, data["FVC_inf"], data["FVC_sup"], alpha=0.5, color="#ffcd3c")
            ax.set_ylabel("FVC")

        f, axes = plt.subplots(1, 3, figsize=(15, 5))
        chart("ID00007637202177411956430", axes[0])
        chart("ID00099637202206203080121", axes[1])
        chart("ID00010637202177584971671", axes[2])
        plt.savefig("./Results/osic/ag_pred_{}.png".format(args.method))