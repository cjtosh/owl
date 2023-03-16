import pandas as pd
import numpy as np
import random
import os
import pickle
import argparse
from owl.regression import LinearRegression
from owl.ball import L1Ball
from scipy.special import xlogy
from time import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing arguments')
    parser.add_argument('--i', type=int, default=1, help="The random seed.")

    start = time()
    ADMMSTEPS = 20000

    epsilons = np.logspace(-4, -1, 50)
    neps = len(epsilons)

    args = parser.parse_args()
    seed = 100 + ((args.i-1) // neps)

    eps_idx = (args.i-1) % neps
    eps = epsilons[eps_idx]

    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_csv("data/microcredit.csv")
    X = df["treatment"].to_numpy()
    X = np.column_stack((X, np.ones(len(X))))
    y = df["profit"].to_numpy()
    weights = df["weight"].to_numpy()
    n = len(y)

    folder = "results/microcredit"
    os.makedirs(folder, exist_ok=True)

    fname = os.path.join(folder, str(eps) + "_" + str(seed) + ".pkl")


    idx_core, = (weights>=1.0).nonzero()
    idx_outliers, = (weights < 1.0).nonzero()

    sidx_core = np.random.choice(idx_core, replace=True, size=len(idx_core))
    sidx_outliers = np.random.choice(idx_outliers, replace=True, size=len(idx_outliers))
    sidx = np.concatenate((sidx_core, sidx_outliers))

    sX = X[sidx,:]
    sy = y[sidx]

    slm = LinearRegression(X=sX, y=sy)
    slm.maximize_weighted_likelihood()

    mle_ate = slm.clf.coef_[0]

    ball = L1Ball(n=n, r=(2*eps))
    slm.fit_owl(ball=ball,admmsteps=ADMMSTEPS, eta=1e-7, admmtol=1e-15, thresh=0.0)
    l1_ate = slm.clf.coef_[0]

    prob = slm.w/np.sum(slm.w)
    okl = np.nansum(xlogy(prob , prob)) - np.dot(prob, slm.log_likelihood())

    e = np.ones(n)/n
    obs_eps = np.sum(np.abs(prob-e))/2.0
    result = {"seed":seed, 
              "Epsilon":eps, 
              "Observed epsilon":obs_eps, 
              "MLE ATE":mle_ate, 
              "OWL ATE":l1_ate, 
              "OKL estimate":okl}

    end = time()

    print("time:", round((end-start), 2))
    with open(fname, 'wb') as io:
        pickle.dump(result, io)

    