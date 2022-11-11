import numpy as np
import argparse
from copy import deepcopy
from bmm import BernoulliMM
from balls_kdes import ProbabilityBall
from cmodels import fit_mle, fit_owl
import os, sys
import pickle

ADMMSTEPS=1000

'''
    This script implements a Bernoulli mixture model simulation.
'''

def pred_cooccurrence(lam: np.ndarray, pi: np.ndarray):
    C_tens = np.einsum('hi, hj -> hij', lam, lam)
    C = np.einsum('hij, h -> ij', C_tens, pi)
    return(C)


def obs_coocur(X, w=None):
    n, p = X.shape
    if w is None:
        w = np.ones(n)/n
    else:
        w = w/np.sum(w)

    observed_cooccurrence = np.zeros((p,p))
    for i in range(n):
        observed_cooccurrence += w[i]*np.multiply.outer(X[i], X[i])
    return(observed_cooccurrence)


def simulated_bmm_data(n, K, p, alpha=0.1, beta=0.1):
    lam = np.random.beta(a=alpha, b=beta, size=(K, p))
#     pi = np.random.dirichlet(np.ones(K))
    pi = np.ones(K)/K
    z = np.random.choice(K, size=n, replace=True, p=pi)
    X = np.empty((n,p), dtype=int)
    for i, k in enumerate(z):
        x = np.random.rand(p)
        X[i] = (x < lam[k]).astype(int)
    return(lam, pi, X, z)

def corrupt(x):
    zero_mask = x == 0
    nzeros = np.sum(zero_mask)
    v = np.random.rand(nzeros)
    x[zero_mask] = (v < 0.5).astype(int)
    return(x)

def simulation(X_, K, epsilon, corr_type, true_C, z_=None, lam_=None):
    X = deepcopy(X_)
    z = deepcopy(z_)
    lam = deepcopy(lam_)

    n, p = X_.shape
    results = []
    n_corrupt = int(epsilon*n)
    
    bmm = BernoulliMM(X=X, K=K, hard=False)
    mle = fit_mle(model=bmm)

    if corr_type=='max':
        lls = mle.log_likelihood_vector() ## Get likelihood values
        inds_corrupt = np.argsort(-lls)[:n_corrupt] ## Corrupt largest indices
    else:
        inds_corrupt = np.random.choice(n, size=n_corrupt, replace=False)

    print("Number of corruptions:", len(inds_corrupt), file=sys.stderr)
    print("Fraction of corruptions:", len(inds_corrupt)/len(z), file=sys.stderr)

    for i in inds_corrupt:
        X[i] = corrupt(X[i])
    
    corrupt_mask = np.isin(np.arange(n), inds_corrupt)
    uncorrupt_mask = ~corrupt_mask

    ## Evaluate uncorrupted MLE
    cooccur_dist = mle.cooccurrence_distance(true_C)
    l1_dist = mle.mean_mae(lam)
    ari = mle.adjusted_rand_index(z, uncorrupt_mask)

    print("Uncorrupted MLE l1 dist:", l1_dist, file=sys.stderr)
    print("Uncorrupted MLE Coccurrence dist:", cooccur_dist, file=sys.stderr)
    
    results.append({"Method": "Uncorrupted MLE", 
                    "Corruption fraction": epsilon, 
                    "Parameter L1 distance": l1_dist,
                    "Co-occurrence L1 distance": cooccur_dist,
                    "Adjusted Rand Index": ari,
                    "Corruption type": corr_type})

    ## Regular MLE
    bmm = BernoulliMM(X=X, K=K, hard=False)
    mle = fit_mle(model=bmm)
    cooccur_dist = mle.cooccurrence_distance(true_C)
    l1_dist = mle.mean_mae(lam)
    ari = mle.adjusted_rand_index(z, uncorrupt_mask)

    results.append({"Method": "MLE", 
                    "Corruption fraction": epsilon, 
                    "Parameter L1 distance": l1_dist,
                    "Co-occurrence L1 distance": cooccur_dist,
                    "Adjusted Rand Index": ari,
                    "Corruption type": corr_type})

    ## TV OWL
    bmm = BernoulliMM(X=X, K=K, hard=True)
    tv_ball = ProbabilityBall(dist_type='l1', n=n, r=epsilon)
    owl_tv = fit_owl(bmm, tv_ball, admmsteps=ADMMSTEPS, verbose=False)
    cooccur_dist = owl_tv.cooccurrence_distance(true_C)
    l1_dist = owl_tv.mean_mae(lam)
    ari = owl_tv.adjusted_rand_index(z, uncorrupt_mask)

    results.append({"Method": "OWL (TV)", 
                    "Corruption fraction": epsilon, 
                    "Parameter L1 distance": l1_dist,
                    "Co-occurrence L1 distance": cooccur_dist,
                    "Adjusted Rand Index": ari,
                    "Corruption type": corr_type})

    return(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing arguments')
    parser.add_argument('--seed', type=int, default=100, help="The random seed.")
    parser.add_argument('--dataset', type=str, default='simul', help="The dataset we're looking at.")
    parser.add_argument('--corr_type', type=str, default='max', help="The random seed.")


    args = parser.parse_args()
    seed = args.seed
    dataset = args.dataset
    corr_type = args.corr_type
    np.random.seed(seed)

    if dataset=='simul':
        n = 1000
        K = 3
        p = 100
        lam, pi, X, z = simulated_bmm_data(n, K, p)
        true_C = pred_cooccurrence(lam=lam, pi=pi)
        epsilons = np.linspace(start=0.01, stop=0.25, num=10)
        
    full_results = []

    for epsilon in epsilons:
        results = simulation(X_=X, K=K, epsilon=epsilon, corr_type=corr_type, true_C=true_C, z_=z, lam_=lam)
        full_results.extend(results)



    folder = "results/bmm_simulation/dim_" + str(p)
    os.makedirs(folder, exist_ok=True)

    fname = os.path.join(folder, corr_type+"_"+str(seed)+".pkl")
    with open(fname, 'wb') as io:
        pickle.dump(full_results, io)

    