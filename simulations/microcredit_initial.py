import pandas as pd
import numpy as np
import random
import os
import pickle
import argparse
from owl.models import fit_owl
from owl.regression import LinearRegression
from owl.ball import L1Ball
from scipy.special import xlogy
from time import time

def tv_dist_to_unif(w):
    n = len(w)
    w = w/np.sum(w)
    e = np.ones(n)/n
    return(np.sum(np.abs(w-e))/2.)

if __name__ == "__main__":
    df = pd.read_csv("../data/micro-credit-profit-vs-treatment-data.csv")
    # Drop empty column
    df = df.drop(df.columns[[0]], axis=1)

    X = df["treatment"].to_numpy()
    X = np.column_stack((X, np.ones(len(X))))
    y = df["profit"].to_numpy()
    n = len(y)

    l1_ball = L1Ball(n=n, r=1.0)
    lm = LinearRegression(X=X, y=y)


    eps_array = np.logspace(-4, -1, 50)

    models, values  = fit_owl(lm, l1_ball,
                          epsilons=2*eps_array,
                          admmsteps=30000,
                          eta=1e-7,
                          admmtol=1e-15,
                          n_workers=6,
                          thresh=0.0,
                          return_all=True)
    eps_array_observed = [tv_dist_to_unif(m.w) for m in models]

    ATEs = np.array([m.clf.coef_[0] for m in models])

    ## Now do the MLE
    lm = LinearRegression(X=X, y=y)
    lm.maximize_weighted_likelihood()
    ATE_mle = lm.clf.coef_[0]

    ## Save out these results
    orig_df = pd.DataFrame({"Epsilon":np.concatenate(([0], eps_array)), 
                            "Observed epsilon": np.concatenate(([0.0], eps_array_observed)),
                            "OWL ATE": np.concatenate(([ATE_mle],ATEs)),
                            "OKL estimate": np.concatenate(([np.nan],values))})


    folder = "results/microcredit"
    os.makedirs(folder, exist_ok=True)
    orig_df.to_csv(os.path.join(folder, "owl_original.csv"))

    ## Now select a model
    ## 1. Filter for monotonicity
    idx = [0]
    for i in range(1, len(eps_array_observed)):
        val = eps_array_observed[i]
        if val > eps_array_observed[idx[-1]]:
            idx.append(i)
    idx = np.array(idx)

    eps_retained = eps_array[idx]
    values_retained = np.array(values)[idx]
    smoothed_values = np.minimum.accumulate(values_retained)


    ## 2. Calculate curvature
    deriv_1 = np.gradient(smoothed_values, eps_retained)
    deriv_2 = np.gradient(deriv_1, eps_retained)
    curv = deriv_2/np.power(1+ np.square(deriv_1), 1.5)

    ## 3. Choose model that maximizes curvature
    j = np.argmax(curv) #The index at which curvature is maximized
    m = models[j] # The model at the optimal value;

    ## Create new dataframe by adding the weights of this model as a column.
    new_df = df.copy()
    new_df['weight'] = m.w

    ## Save out the new dataframe
    new_df.to_csv("../data/microcredit.csv")







