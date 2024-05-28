import os
import pickle
import numpy as np
import argparse
import scipy.stats as stats
from scipy.special import xlogy
from owl.ball import L1Ball
from owl.kde import RBFKDE, knn_bandwidth
from owl.gaussian import Gaussian
from tqdm import tqdm


def gaussian_corruption_comparison(X_:np.ndarray, mu_:np.ndarray, cov_:np.ndarray, epsilon:float, scale:float, corr_type:str):
    results = []
    mu = mu_.copy()
    cov = cov_.copy()

    g = Gaussian(X=X_)
    n = g.n
    p = g.p
    X = g.X.copy()
    n_corrupt = int(epsilon*n)

    g.maximize_weighted_likelihood()
    mu_dist = np.mean(np.square(mu - g.mu ))

    results.append({"Method": "Uncorrupted MLE", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mu_dist,
                    "Corruption type": corr_type})

    if corr_type=='max':
        lls = g.log_likelihood() ## Get likelihood values
        inds_corrupt = np.argsort(-lls)[:n_corrupt] ## Corrupt largest indices
    else:
        inds_corrupt = np.random.choice(n, size=n_corrupt, replace=False)
    
    for idx in inds_corrupt:
        X[idx,:] = np.random.uniform(low=-scale, high=scale, size=X[idx,:].shape)


    ## MLE
    g = Gaussian(X=X)
    g.maximize_weighted_likelihood()
    mu_dist = np.mean(np.square(mu - g.mu ))
    results.append({"Method": "MLE", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mu_dist,
                    "Corruption type": corr_type})
    

    weights = {}
    weights["inds_corrupt"] = inds_corrupt

    ## OWL - TV
    g = Gaussian(X=X)
    tv_ball = L1Ball(n=n, r=2*epsilon)
    g.fit_owl(ball=tv_ball, n_iters=10)
    prob = g.w/np.sum(g.w)
    mu_dist = np.mean(np.square(mu - g.mu ))
    results.append({"Method": "OWL (TV)", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mu_dist,
                    "Corruption type": corr_type})
    
    weights["OWL (TV)"] = prob

    ## OWL - Kernelized TV
    best_ll = -np.infty
    hell_dist_sel = None
    mu_dist_sel = None
    for k in [5, 10, 25, 50]:
        g = Gaussian(X=X)
        tv_ball = L1Ball(n=n, r=2*epsilon)
        bandwidth = knn_bandwidth(X, k)
        kde = RBFKDE(X, bandwidth)
        g.fit_owl(ball=tv_ball, n_iters=10, kde=kde)
        prob = g.w/np.sum(g.w)
        ll = np.dot(prob, g.log_likelihood()) - np.nansum(xlogy(prob , prob))
        mu_dist = np.mean(np.square(mu - g.mu ))

        
        results.append({"Method": "OWL (Kernelized, k=" + str(k) + ")", 
                        "Corruption fraction": epsilon, 
                        "Mean MSE": mu_dist,
                        "Corruption type": corr_type})
        
        weights["OWL (Kernelized, k=" + str(k) + ")"] = prob

        if ll > best_ll:
            best_ll = ll
            mu_dist_sel = mu_dist

    results.append({"Method": "OWL (Kernelized, adaptive)", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mu_dist_sel,
                    "Corruption type": corr_type})

    return(results, weights)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing arguments')
    parser.add_argument('--seed', type=int, default=100, help="The random seed.")
    parser.add_argument('--scale', type=float, default=5.0, help="The scale of corruption.")
    parser.add_argument('--dim', type=int, default=2, help="The dimension of the problem.")
    parser.add_argument('--n', type=int, default=100, help="The number of data points.")
    parser.add_argument('--corr_type', type=str, default='max', help="The type of corruption.")

    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    scale = args.scale
    dim = args.dim
    n = args.n
    corr_type = args.corr_type

    folder = os.path.join("results/gaussian", "dim_"+str(dim))
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, corr_type+"_"+str(seed)+".pkl")
    weight_fname = os.path.join(folder, "weights_"+corr_type+"_"+str(seed)+".pkl")

    mu = np.random.uniform(low=-scale, high=scale, size=dim)
    cov = np.eye(dim)
    X = stats.multivariate_normal.rvs(mean=mu, cov=cov, size=n, random_state=None)

    full_results = []
    full_weights = {}
    epsilons = np.linspace(start=0.01, stop=0.5, num=15)
    for epsilon in tqdm(epsilons):
        results, weights = gaussian_corruption_comparison(X, mu, cov, epsilon, scale, corr_type)
        full_results.extend(results)
        full_weights[epsilon] = weights
        with open(fname, 'wb') as io:
            pickle.dump(full_results, io)

        with open(weight_fname, 'wb') as io:
            pickle.dump(full_weights, io)
    
   

    