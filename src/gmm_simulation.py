import numpy as np
import argparse
from copy import deepcopy
from spherical_gmm import SphericalGMM
from cmodels import fit_mle, fit_owl, fit_kernelized_owl
from balls_kdes import ProbabilityBall, KDE, knn_bandwidth
import os, sys
import pickle
from tqdm import tqdm

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
    
    gmm = SphericalGMM(X, K=K, hard=False)

    ## Fit a MLE
    mle = fit_mle(gmm)

    ## Uncorrupted distances
    mean_dist = mle.mean_mse(mu)
    hell_dist = mle.hellinger_distance(mu, tau)

    # print("Uncorrupted MLE mean dist:", mean_dist, file=sys.stderr)
    # print("Uncorrupted MLE Hellinger dist:", hell_dist, file=sys.stderr)

    if corr_type=='max':
        lls = mle.log_likelihood_vector() ## Get likelihood values
        inds_corrupt = np.argsort(-lls)[:n_corrupt] ## Corrupt largest indices
    else:
        inds_corrupt = np.random.choice(n, size=n_corrupt, replace=False)

    # print("Number of corruptions:", len(inds_corrupt))
    # print("Fraction of corruptions:", len(inds_corrupt)/len(z))
    for i in inds_corrupt:
        idxs = np.random.choice(p, size=int(0.5*p), replace=False)
        X[i][idxs] = corr_scale*np.random.choice([-1., 1.], size=len(idxs), replace=True)
    
    
    corrupt_mask = np.isin(np.arange(n), inds_corrupt)
    uncorrupt_mask = ~corrupt_mask

    ari = mle.adjusted_rand_index(z, uncorrupt_mask)
    results.append({"Method": "Uncorrupted MLE", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mean_dist,
                    "Hellinger distance": hell_dist,
                    "Adjusted Rand Index": ari,
                    "Corruption type": corr_type,
                    "Corruption scale": corr_scale})

    ## MLE on corrupted data
    gmm = SphericalGMM(X, K=K, hard=False)

    mle = fit_mle(gmm)
    mean_dist = mle.mean_mse(mu)
    hell_dist = mle.hellinger_distance(mu, tau)
    ari = mle.adjusted_rand_index(z, uncorrupt_mask)

    results.append({"Method": "MLE", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mean_dist,
                    "Hellinger distance": hell_dist,
                    "Adjusted Rand Index": ari,
                    "Corruption type": corr_type,
                    "Corruption scale": corr_scale})

    ## OWL with TV dist
    gmm = SphericalGMM(X, K=K, hard=True)
    tv_ball = ProbabilityBall(dist_type='l1', n=n, r=epsilon)
    owl_tv = fit_owl(gmm, tv_ball, admmsteps=ADMMSTEPS, verbose=False)

    mean_dist = owl_tv.mean_mse(mu)
    hell_dist = owl_tv.hellinger_distance(mu, tau)
    ari = owl_tv.adjusted_rand_index(z, uncorrupt_mask)

    results.append({"Method": "OWL (TV)", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mean_dist,
                    "Hellinger distance": hell_dist,
                    "Adjusted Rand Index": ari,
                    "Corruption type": corr_type,
                    "Corruption scale": corr_scale})

    ## Kernelized OWL with TV dist
    gmm = SphericalGMM(X, K=K, hard=True)
    bandiwdth_schedule = [knn_bandwidth(X, k) for k in [5, 10, 30, 50]]
    kde = KDE(X=X, bandwidth=bandiwdth_schedule[0], method='rbf')
    owl_kern_tv = fit_kernelized_owl(gmm, tv_ball, kde, bandiwdth_schedule, repeats=3, admmsteps=ADMMSTEPS, verbose=False)
    mean_dist = owl_kern_tv.mean_mse(mu)
    hell_dist = owl_kern_tv.hellinger_distance(mu, tau)
    ari = owl_kern_tv.adjusted_rand_index(z, uncorrupt_mask)

    results.append({"Method": "OWL (TV-Kernelized)", 
                    "Corruption fraction": epsilon, 
                    "Mean MSE": mean_dist,
                    "Hellinger distance": hell_dist,
                    "Adjusted Rand Index": ari,
                    "Corruption type": corr_type,
                    "Corruption scale": corr_scale})

    return(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing arguments')
    parser.add_argument('--seed', type=int, default=100, help="The random seed.")
    parser.add_argument('--corr_scale', type=float, default=5.0, help="The random seed.")
    parser.add_argument('--corr_type', type=str, default='max', help="The random seed.")

    args = parser.parse_args()
    seed = args.seed

    np.random.seed(seed)
    corr_type = args.corr_type
    corr_scale = args.corr_scale

    full_results = []

    n = 1000
    K = 3
    p = 10
    stdv_0 = 2.0
    stdv = 0.5
    mu, stdvs, pi, X, z = simulated_gmm_data(p=p, n=n, K=K, stdv_0=stdv_0, stdv=stdv)
    epsilons = np.linspace(start=0.01, stop=0.25, num=10)

    for epsilon in tqdm(epsilons):
        results = simulation(X_=X, mu_=mu, stdvs_=stdvs, z_=z, K=K, epsilon=epsilon, corr_type=corr_type, corr_scale=corr_scale)
        full_results.extend(results)



    folder = "results/gmm_simulation/dim_" + str(p)
    os.makedirs(folder, exist_ok=True)

    fname = os.path.join(folder, corr_type+"_"+str(seed)+".pkl")
    with open(fname, 'wb') as io:
        pickle.dump(full_results, io)

    