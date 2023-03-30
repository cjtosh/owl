import numpy as np
import argparse
from copy import deepcopy
from owl.models import fit_owl
from owl.mixture_models import SphericalGMM
from owl.ball import L1Ball
from tqdm import tqdm
import os
import pickle


ADMMSTEPS = 5000

def simulated_gmm_data(p, n, K, stdv_0, stdv):
    mu =  stdv_0 * np.random.randn(K,p)
    # stdvs = np.random.uniform(low=0.0, high=stdv, size=K)
    stdvs = np.ones(K)*stdv
    # pi = np.random.dirichlet(np.ones(K))
    pi = np.ones(K)/K
    
    X = np.empty((n,p))
    z = np.random.choice(K, size=n, replace=True, p=pi)
    for i, k in enumerate(z):
        X[i] = mu[k] + np.random.normal(loc=0.0, scale=stdvs[k], size=p)
    
    return(mu, stdvs, pi, X, z)

def simulation(X_, mu_, stdvs_, z_, K, epsilon, corr_type, corr_scale):
    X = deepcopy(X_)
    z = deepcopy(z_)
    mu = deepcopy(mu_)
    stdvs = deepcopy(stdvs_)
    tau = 1.0/np.square(stdvs)

    results = []
    n_corrupt = int(epsilon*n)
    
    ## Fit a MLE
    mle = SphericalGMM(X, K=K, repeats=100, hard=False)
    mle.fit_mle()
    
    ## Uncorrupted distances
    mean_dist = mle.mean_mse(mu)
    
    ## Corrupt the data
    if corr_type=='max':
        lls = mle.log_likelihood() ## Get likelihood values
        inds_corrupt = np.argsort(-lls)[:n_corrupt] ## Corrupt largest indices
    else:
        inds_corrupt = np.random.choice(n, size=n_corrupt, replace=False)

    for i in inds_corrupt:
        idxs = np.random.choice(p, size=int(0.5*p), replace=False)
        X[i][idxs] = corr_scale*np.random.choice([-1., 1.], size=len(idxs), replace=True)

    results.append({"Method": "Uncorrupted MLE", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mean_dist,
                    "Corruption type": corr_type,
                    "Corruption scale": corr_scale})

    ## MLE on corrupted data
    mle = SphericalGMM(X, K=K, repeats=100, hard=False)
    mle.fit_mle()
    mean_dist = mle.mean_mse(mu)

    results.append({"Method": "MLE", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mean_dist,
                    "Corruption type": corr_type,
                    "Corruption scale": corr_scale})



    ## OWL with TV dist (Search for radius)
    gmm = SphericalGMM(X, K=K, hard=True)
    l1_ball = L1Ball(n=n, r=1.0)
    owl_tv = fit_owl(gmm, 
                     l1_ball, 
                     epsilons=np.linspace(0.01, 0.5, 20), 
                     admmsteps=ADMMSTEPS,
                     n_workers=6)
    mean_dist = owl_tv.mean_mse(mu)
    
    results.append({"Method": "OWL (TV)", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mean_dist,
                    "Corruption type": corr_type,
                    "Corruption scale": corr_scale})
    

    ## OWL with TV dist (known radius)
    owl_tv = SphericalGMM(X, K=K, hard=True)
    l1_ball = L1Ball(n=n, r=epsilon)
    owl_tv.fit_owl(l1_ball, admmsteps=ADMMSTEPS)
    mean_dist = owl_tv.mean_mse(mu)

    results.append({"Method": "OWL ($\epsilon$ known)", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mean_dist,
                    "Corruption type": corr_type,
                    "Corruption scale": corr_scale})

    return(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gaussian mixture model simulations')
    parser.add_argument('--seed', type=int, default=100, help="The random seed.")
    parser.add_argument('--n', type=int, default=1000, help="The number of samples.")
    parser.add_argument('--p', type=int, default=10, help="The dimension of the problem.")
    parser.add_argument('--stdv', type=float, default=0.5, help="The scale of corruptions.")
    parser.add_argument('--corr_scale', type=float, default=5.0, help="The scale of corruptions.")
    parser.add_argument('--corr_type', type=str, default='max', help="The method of choosing corruptions (input 'max' or 'rand').")
    args = parser.parse_args()
    seed = args.seed

    np.random.seed(seed)
    corr_type = args.corr_type
    corr_scale = args.corr_scale
    n = args.n
    p = args.p
    stdv = args.stdv
    full_results = []

    K = 3
    stdv_0 = 2.0
    mu, stdvs, pi, X, z = simulated_gmm_data(p=p, n=n, K=K, stdv_0=stdv_0, stdv=stdv)
    epsilons = np.linspace(start=0.01, stop=0.25, num=10)

    folder = "results/gmm_simulation/dim_" + str(p)
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, corr_type+"_"+str(seed)+".pkl")

    for epsilon in tqdm(epsilons):
        results = simulation(X_=X, mu_=mu, stdvs_=stdvs, z_=z, K=K, epsilon=epsilon, corr_type=corr_type, corr_scale=corr_scale)
        full_results.extend(results)
        with open(fname, 'wb') as io:
            pickle.dump(full_results, io)


    

    
    

    