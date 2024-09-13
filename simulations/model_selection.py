from scipy.stats import norm
from owl.mixture_models import GeneralGMM
from owl.kde import RBFKDE
from owl.ball import L1Ball
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    os.makedirs("./figures/model_selection", exist_ok=True)

    #Let us fit a mixture of 5 Gaussians to this data both using MLE and 0.05 TV corruption.
    K = 3 
    epsilons = [0.05, 0.1]

    # Set the parameters for the mixture components
    mu1 = 0  # mean of the first component
    sigma1 = 1  # standard deviation of the first component

    mu2 = 5  # mean of the second component
    sigma2 = 1  # standard deviation of the second component
    #alpha2s = [0,5,10,15]  # skewness parameter of the second component
    df = 10

    # Generate random samples from the mixture distribution
    n = 1000  # number of samples
    weights = [0.25, 0.75]  # weights of the mixture components
    knn = np.log2(n).astype(np.int64)

    Ks = [1,2,3,4,5]

    np.random.seed(12345)
    # Generate random indices to select the component for each sample
    component_indices = np.random.choice([0, 1], size=n, p=weights)

    # Generate data from the mixture distribution

    scaled_chi2 = (np.random.chisquare(df,size=n)-df)/np.sqrt(2*df)
    data = (1-component_indices)*np.random.normal(loc=mu1, scale=sigma1, size=n) + component_indices*sigma2*(mu2 + scaled_chi2)
        
    X = data.reshape((-1,1))

    def run_methods(X, Ks, thread_print=print):
        kde=RBFKDE(X, neighbors=knn)
        fits = [None for K in Ks]
        for j, K in enumerate(Ks):
            thread_print(f"K={K}")
            d = {}
            # First fit the MLE using EM algorithm from GMMs.
            mle = GeneralGMM(X=X, K=K, repeats=10, hard=False)
            mle.fit_mle()
            d['MLE (soft EM)'] = mle

            mle_hard = GeneralGMM(X=X, K=K, repeats=10, hard=True)
            mle_hard.fit_mle()
            d['MLE (hard EM)'] = mle_hard

            for eps in epsilons:
                thread_print(f'eps = {eps}')
                rob = GeneralGMM(X=X, K=K, repeats=5, hard=True)
                l1ball = L1Ball(n=n, r=2*eps)
                rob.fit_owl(ball=l1ball, n_iters=10, kde=kde)
                d[f'OWL ε={eps}'] = rob

            fits[j] = d
        return fits

    fits=run_methods(X, Ks)

    # Compute a weighted AIC/BIC criteria for 1D Gaussian mixture models 

    def criteria_aic(model_fit):
        ll = sum(model_fit.log_likelihood()*model_fit.w)
        k = 2*model_fit.K # The total number of parameters in the model
        return (2*k - 2*ll)

    def criteria_bic(model_fit):
        ll = sum(model_fit.log_likelihood()*model_fit.w)
        k = 2*model_fit.K # The total number of parameters in the model
        n = model_fit.n
        return (k*np.log(n) - 2*ll)


    fit_df = pd.DataFrame(fits)

    aic_scores = fit_df.applymap(criteria_aic)
    bic_scores = fit_df.applymap(criteria_bic)


    n_methods=4
    colors = sns.color_palette("Dark2", n_methods)

    # Plot AIC scores
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for i, column in enumerate(aic_scores.columns):
        axs[0].set_ylabel('AIC')
        axs[0].set_xlabel('Number of components (K)')
        axs[1].set_ylabel('BIC')
        axs[1].set_xlabel('Number of components (K)')
        line_aic, = axs[0].plot(Ks, aic_scores[column], label=f'{column}', color=colors[i])
        line_bic, = axs[1].plot(Ks, bic_scores[column], label=f'{column}', color=colors[i])

        # Find the minimum AIC and BIC points
        min_aic_idx = aic_scores[column].idxmin()
        min_bic_idx = bic_scores[column].idxmin()
        
        # Annotate the minimum points with a big dot
        axs[0].plot(Ks[min_aic_idx], aic_scores[column][min_aic_idx], 'o', markersize=5, markerfacecolor='none', color=line_aic.get_color())
        axs[1].plot(Ks[min_bic_idx], bic_scores[column][min_bic_idx], 'o', markersize=5, markerfacecolor='none', color=line_bic.get_color())

    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper right', bbox_to_anchor=(.9,.9))
    # Display the plot
    plt.suptitle("Selecting the number of mixture components")
    plt.savefig("../figures/model_selection/num_components.pdf", bbox_inches='tight')

    import math

    def plot_gaussian_components(x, means, std_devs, fracs, linestyle='-', label='Gaussian Components',
            ax=plt, color='black', thresh=0.01):
        """Plot each Gaussian component of the mixture model."""
        for i, (frac, mean, std_dev) in enumerate(zip(fracs, means, std_devs)):
            if frac > thresh:
                ax.plot(x, frac * norm.pdf(x, mean, std_dev), linestyle=linestyle, color=color)
        # Add a single legend entry for the components
        ax.plot([], [], linestyle=linestyle, color=color, label=label)

    n_methods = fit_df.shape[1]
    # The k-1 is because idxmins are for some reasons following numbering at 1
    best_fits = [ (col, fit_df.at[k, col]) for col, k in bic_scores.idxmin().items()]

    fig, axs = plt.subplots(math.ceil(n_methods/2), 2, constrained_layout=True, sharey='row', sharex='col', figsize=(6,3))


    x = np.linspace(np.min(X), np.max(X), 1000)

    handles = []
    labels = []

    legendObs = True

    for i, ax in enumerate(axs.flat):

        name, fit = best_fits[i]
        # Plot range
        if not name.startswith('MLE'):
            ax.hist(X, bins=50, weights=fit.w, density=True, 
                            histtype='step', 
                            label=f'Re-weighted', 
                            color=colors[i])
            
                # Plot the histogram of the observed data
            ax.hist(X, bins=50, density=True, histtype='stepfilled', 
                            color='gray', alpha=0.2)
        else:
            ax.hist(X, bins=50, density=True, histtype='stepfilled', 
                        label='Observed' if legendObs else None, color='gray', alpha=0.2)
            legendObs=False

        # Plot each Gaussian component
        plot_gaussian_components(x, means=fit.mu.flatten(),
                                    std_devs=1/np.sqrt(fit.prec_mats.flatten()), 
                                    fracs=fit.cluster_weights/np.sum(fit.cluster_weights),
                                    label=name,
                                    linestyle='--',
                                    color=colors[i],
                                    ax=ax)
        #ax.legend()
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l


        

    order = [1,0,4,3,6,5,2]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower left', framealpha=1, ncol=4,
            bbox_to_anchor=(0,1.005))
    # Display the plot
    plt.tight_layout()
    plt.savefig("../figures/model_selection/reweighted_hists.pdf", bbox_inches='tight')
