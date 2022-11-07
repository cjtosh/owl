import abc
import numpy as np
from scipy.spatial.distance import cdist
from copy import deepcopy
from scipy.stats import norm
from gmm import cluster_statistics

class FSRobustModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, n:int, w:np.ndarray=None, **kwargs):
        self.n = n
        self.w = w
        if w is None:
            self.w = np.ones(self.n)

    @abc.abstractmethod
    def EM_step(self, n_steps:int=1, **kwargs) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def cdf_vector(self, **kwargs) -> np.ndarray:
        raise NotImplementedError


    def fsrobust_mle(self, p:float, n_iters:int, emsteps:int=20,  **kwargs) -> None:
        ## Alternate between EM and setting weights
        for _ in range(n_iters):
            ## Run EM
            self.EM_step(n_steps=emsteps)

            ## Get the cdf values for each point
            cdf_scores = self.cdf_vector()

            ## weight things within the various percentiles
            self.w = np.ones(self.n)

            high_mask = cdf_scores > (1-p)
            self.w[high_mask] = (1.0 - cdf_scores[high_mask])/p

            low_mask = cdf_scores < p
            self.w[low_mask] =cdf_scores[low_mask]/p


class FSRobustGMM(FSRobustModel): ## 1d
    def __init__(self, X:np.ndarray, K:int, w:np.ndarray=None):
        n = len(X)
        super().__init__(n=n, w=w)
        self.X = X
        self.K = K

        inds = np.random.choice(self.n, size=self.K, replace=False).astype(int)
        self.mu = X[inds]
        self.tau = np.ones(self.K)
        self.pi = np.ones(self.K)/self.K
        self.z = np.random.choice(self.K, size=self.n, replace=True).astype(int)

    def EM_step(self, n_steps:int=1):
        for _ in range(n_steps):
            ## Solve for optimal z's (Hard E-step)
            square_dists = cdist(self.X[:,np.newaxis], self.mu[:,np.newaxis], metric='sqeuclidean')
            prec_prod = np.einsum('i,h -> ih', self.w, self.tau)
            log_probs = 0.5*np.log(prec_prod) - 0.5*prec_prod*square_dists


            self.z = np.argmax(log_probs, axis=1)
            self.cluster_to_inds, self.cluster_weights, self.cluster_weighted_means, self.cluster_weighted_sum_of_squares = cluster_statistics(self.X[:,np.newaxis], self.z, self.K, self.w)

            ## Maximize parameters given optimal z's (Hard M-step)
            self.mu = np.squeeze(self.cluster_weighted_means)
            cluster_sizes = np.array([len(cluster) for cluster in self.cluster_to_inds.values()])
            self.tau = cluster_sizes/self.cluster_weighted_sum_of_squares
            self.pi = cluster_sizes/self.n

    def cdf_vector(self):
        stdvs = np.sqrt(1./self.tau)
        cdf_scores = np.zeros(self.n)
        for pi, mu, stdv in zip(self.pi, self.mu, stdvs):
            cdf_scores =  cdf_scores + pi*norm.cdf(self.X, loc=mu, scale=stdv)

        return(cdf_scores)

    def log_likelihood_vector(self):
        w_prec = self.w*self.tau[self.z]
        return(0.5*np.log(w_prec) - 0.5*w_prec*np.sum(np.square(self.X - self.mu[self.z]), axis=1))



def gmm_random_reruns(X:np.ndarray, K:int, n_reruns:int, p:float, n_iters:int, emsteps:int=20):
    current_model = FSRobustGMM(X, K)
    current_weighted_ll = np.sum(current_model.log_likelihood_vector())
    for _ in range(n_reruns):
        model = FSRobustGMM(X, K)
        model.fsrobust_mle(p=p, n_iters=n_iters, emsteps=emsteps)
        

class FSRobustGaussian(FSRobustModel): ## 1d
    def __init__(self, X:np.ndarray, w:np.ndarray=None):
        n = len(X)
        super().__init__(n=n, w=w)
        self.X = X

        self.mu = np.mean(X)
        self.var = np.var(X)

    def EM_step(self, n_steps:int=1):
        w_sum = np.sum(self.w)
        self.mu = np.sum(self.w * self.X)/w_sum
        self.var = np.sum( self.w * np.square(self.X - self.mu ))/w_sum

    def cdf_vector(self):
        stdv = np.sqrt(self.var)
        cdf_scores =norm.cdf(self.X, loc=self.mu, scale=stdv)
        return(cdf_scores)

    def log_likelihood_vector(self):
        stdv = np.sqrt(self.var)
        result = self.w*norm.logpdf(self.X, loc=self.mu, scale=stdv)
        return(result)


