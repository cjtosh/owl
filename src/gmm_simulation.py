import numpy as np
import argparse
from copy import deepcopy
from spherical_gmm import SphericalGMM
from cmodels import fit_mle, fit_owl
from balls_kdes import ProbabilityBall, KDE, knn_bandwidth
import os, sys
import pickle
from scipy.special import xlogy

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


def fit_mmd_owl(X, K, epsilon, repeats=5):
    n, _ = X.shape
    gmm = SphericalGMM(X, K=K)
    best_gmm = SphericalGMM(X, K=K)
    best_ll = -np.infty
    for k in [5, 15, 30, 50]:
        bandwidth = knn_bandwidth(X, k)
        kde = KDE(X=X, bandwidth=bandwidth, method='rbf')
        mmd_ball = ProbabilityBall(dist_type='mmd', n=n, r=epsilon**2, kernel_matrix=kde.mmd_matrix())
        gmm_mmd = fit_owl(gmm, mmd_ball, admmsteps=ADMMSTEPS, repeats=repeats, verbose=False)

        prob = gmm_mmd.w/np.sum(gmm_mmd.w)
        ll = np.dot(prob, gmm_mmd.log_likelihood_vector()) - np.nansum(xlogy(prob , prob))
        if ll > best_ll:
            best_ll = ll
            best_gmm = deepcopy(gmm_mmd)
    return(best_gmm)

def simulation(X_, mu_, stdvs_, z_, K, epsilon, corr_type, corr_scale, use_mmd=True):
    X = deepcopy(X_)
    z = deepcopy(z_)
    mu = deepcopy(mu_)
    stdvs = deepcopy(stdvs_)
    tau = 1.0/np.square(stdvs)

    results = []
    n_corrupt = int(epsilon*n)
    
    gmm = SphericalGMM(X, K=K)

    ## Fit a MLE
    mle = fit_mle(gmm)

    ## Uncorrupted distances
    mean_dist = mle.mean_mse(mu)
    hell_dist = mle.hellinger_distance(mu, tau)

    print("Uncorrupted MLE mean dist:", mean_dist, file=sys.stderr)
    print("Uncorrupted MLE Hellinger dist:", hell_dist, file=sys.stderr)


    if corr_type=='max':
        lls = mle.log_likelihood_vector() ## Get likelihood values
        inds_corrupt = np.argsort(-lls)[:n_corrupt] ## Corrupt largest indices
    else:
        inds_corrupt = np.random.choice(n, size=n_corrupt, replace=False)

    print("Number of corruptions:", len(inds_corrupt))
    print("Fraction of corruptions:", len(inds_corrupt)/len(z))
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
    gmm = SphericalGMM(X, K=K)

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

    if use_mmd:
        owl_mmd = fit_mmd_owl(X, K, epsilon)
        mean_dist = owl_mmd.mean_mse(mu)
        hell_dist = owl_mmd.hellinger_distance(mu, tau)
        ari = owl_mmd.adjusted_rand_index(z, uncorrupt_mask)

        results.append({"Method": "OWL (MMD)", 
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
    parser.add_argument('--drop_mmd', action=argparse.BooleanOptionalAction, help="Do we generate the umap plots?")


    args = parser.parse_args()
    seed = args.seed

    np.random.seed(seed)
    corr_type = args.corr_type
    corr_scale = args.corr_scale
    use_mmd = not args.drop_mmd

    full_results = []

    n = 1000
    K = 3
    p = 10
    stdv_0 = 2.0
    stdv = 0.5
    mu, stdvs, pi, X, z = simulated_gmm_data(p=p, n=n, K=K, stdv_0=stdv_0, stdv=stdv)

    for epsilon in np.linspace(start=0.01, stop=0.25, num=10):
        results = simulation(X_=X, mu_=mu, stdvs_=stdvs, z_=z, K=K, epsilon=epsilon, corr_type=corr_type, corr_scale=corr_scale, use_mmd=use_mmd)
        full_results.extend(results)



    folder = "results/gmm_simulation/dim_" + str(p)
    os.makedirs(folder, exist_ok=True)

    fname = os.path.join(folder, corr_type+"_"+str(seed)+".pkl")
    with open(fname, 'wb') as io:
        pickle.dump(full_results, io)

    