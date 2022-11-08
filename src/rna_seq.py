import pandas as pd
import numpy as np
import random
import os
import pickle
import argparse
from sklearn.decomposition import PCA
from general_gmm import GeneralGMM
from balls_kdes import ProbabilityBall
from cmodels import fit_owl
from sklearn.metrics import adjusted_rand_score
from scipy.special import xlogy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing arguments')
    parser.add_argument('--seed', type=int, default=100, help="The random seed.")

    epsilons = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    neps = len(epsilons)

    args = parser.parse_args()
    seed = 100 + (args.seed // neps)

    eps_idx = args.seed % neps
    eps = epsilons[eps_idx] 

    np.random.seed(seed)
    random.seed(seed)

    X_proc = pd.read_csv("data/cell_data.csv", index_col=0)
    groups = np.array([g.split('__')[1] for g in X_proc.index.values])

    groups_no_b = np.array([x.split('_')[0] for x in groups])
    ngroups_no_b = len(np.unique(groups_no_b))

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_proc)

    folder = "results/rna_seq"
    os.makedirs(folder, exist_ok=True)

    fname = os.path.join(folder, str(eps) + "_" + str(seed) + ".pkl")
    results = []
    for k in np.arange(2, 15):
        gmm_tv = GeneralGMM(X=X_pca, K=k)

        l1_ball = ProbabilityBall(dist_type='l1', n=X_pca.shape[0], r=eps)
        gmm_tv = fit_owl(gmm_tv, l1_ball, kde=None, repeats=5, admmsteps=2000, verbose=False)
        mask = (gmm_tv.w >= 1.0)
        ll_vec = gmm_tv.log_likelihood_vector()
        wll = np.dot(gmm_tv.w,ll_vec)

        probs = gmm_tv.w/np.sum(gmm_tv.w)
        kl = np.nansum(xlogy(probs , probs)) - np.dot(probs, ll_vec)
        num_inliers = np.sum(mask)

        results.append({"K": k, 
                        "epsilon": eps, 
                        "Weighted Log-likelihood": wll,
                        "KL divergence": kl,
                        "Number of inliers": num_inliers,
                        "Adjusted Rand Index": adjusted_rand_score(groups_no_b, gmm_tv.z),
                        "Adjusted Rand Index (with batches)":  adjusted_rand_score(groups, gmm_tv.z),
                        "Adjusted Rand Index (subset)":  adjusted_rand_score(groups_no_b[mask], gmm_tv.z[mask]),
                        "Adjusted Rand Index (subset, with batches)":  adjusted_rand_score(groups[mask], gmm_tv.z[mask])
                        })
            
    with open(fname, 'wb') as io:
        pickle.dump(results, io)

    