import os
import pickle
import numpy as np
import argparse
import scipy.stats as stats
from scipy.special import xlogy
from balls_kdes import ProbabilityBall, KDE, knn_bandwidth
from gaussian import Gaussian
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

    g.EM_step()
    hell_dist = g.hellinger(mu, cov)
    mu_dist = np.mean(np.square(mu - g.mu ))

    results.append({"Method": "Uncorrupted MLE", 
                    "Corruption fraction": epsilon, 
                    "Hellinger distance": hell_dist,
                    "Mean MSE": mu_dist,
                    "Corruption type": corr_type})

    if corr_type=='max':
        lls = g.log_likelihood_vector() ## Get likelihood values
        inds_corrupt = np.argsort(-lls)[:n_corrupt] ## Corrupt largest indices
    else:
        inds_corrupt = np.random.choice(n, size=n_corrupt, replace=False)
    
    for idx in inds_corrupt:
        X[idx,:] = np.random.uniform(low=-scale, high=scale, size=X[idx,:].shape)


    ## MLE
    g = Gaussian(X=X)
    g.EM_step()
    hell_dist = g.hellinger(mu, cov)
    mu_dist = np.mean(np.square(mu - g.mu ))
    results.append({"Method": "MLE", 
                    "Corruption fraction": epsilon, 
                    "Hellinger distance": hell_dist,
                    "Mean MSE": mu_dist,
                    "Corruption type": corr_type})
    

    ## OWL - TV
    g = Gaussian(X=X)
    l1_ball = ProbabilityBall(n=n, dist_type='l1', r=2*epsilon)
    g.am_robust(ball=l1_ball, n_iters=10)
    hell_dist = g.hellinger(mu, cov)
    mu_dist = np.mean(np.square(mu - g.mu ))
    results.append({"Method": "OWL (TV)", 
                    "Corruption fraction": epsilon, 
                    "Hellinger distance": hell_dist,
                    "Mean MSE": mu_dist,
                    "Corruption type": corr_type})
    

    ## OWL - Kernelized TV
    best_ll = -np.infty
    hell_dist = None
    mu_dist = None
    selected_k = None
    for k in [5, 10, 25, 50]:
        g = Gaussian(X=X)
        l1_ball = ProbabilityBall(n=n, dist_type='l1', r=2*epsilon)
        bandwidth = knn_bandwidth(X, k)
        kde = KDE(X, bandwidth)
        g.am_robust(ball=l1_ball, n_iters=10, kde=kde)
        prob = g.w/np.sum(g.w)
        ll = np.dot(prob, g.log_likelihood_vector()) - np.nansum(xlogy(prob , prob))
        if ll > best_ll:
            best_ll = ll
            hell_dist = g.hellinger(mu, cov)
            mu_dist = np.mean(np.square(mu - g.mu ))
            selected_k = k
    
    results.append({"Method": "OWL (Kernelized - TV)", 
                    "Corruption fraction": epsilon, 
                    "Hellinger distance": hell_dist,
                    "Mean MSE": mu_dist,
                    "Corruption type": corr_type})

    return(results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing arguments')
    parser.add_argument('--seed', type=int, default=100, help="The random seed.")
    parser.add_argument('--scale', type=float, default=5.0, help="The scale of corruption.")
    parser.add_argument('--dim', type=int, default=1, help="The dimension of the problem.")
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

    mu = np.random.uniform(low=-scale, high=scale, size=dim)
    cov = np.eye(dim)
    X = stats.multivariate_normal.rvs(mean=mu, cov=cov, size=n, random_state=None)

    full_results = []
    for epsilon in tqdm(np.linspace(start=0.01, stop=0.5, num=15)):
        results = gaussian_corruption_comparison(X, mu, cov, epsilon, scale, corr_type)
        full_results.extend(results)
        with open(fname, 'wb') as io:
            pickle.dump(full_results, io)
    
   

    