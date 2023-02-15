import numpy as np
from owl.models import OWLModel
import scipy.stats as stats
from copy import deepcopy


class Gaussian(OWLModel):
    def __init__(self, 
        X: np.ndarray, ## Input samples [n samples x p features]
        w:np.ndarray = None, ## Weights over the samples (set to None for uniform)
        **kwargs
        ):
        self.X = deepcopy(X)
        if len(X.shape) == 1:
            self.X = self.X[:,np.newaxis]
        n, self.p = X.shape
        super().__init__(n=n, w=w, **kwargs)

        self.mu = np.zeros(self.p)
        self.cov = np.cov(self.X, rowvar=False, ddof=0)

    def maximize_weighted_likelihood(self, **kwargs):
        self.mu = np.average(self.X, axis=0, weights=self.w)
        self.cov = np.cov(self.X, rowvar=False, ddof=0, aweights=self.w)

    def log_likelihood_vector(self):
        tol = 1e-10
        cov = deepcopy(self.cov)
        lam = None
        V = None
        while True:
            try:
                return(stats.multivariate_normal.logpdf(self.X, mean=self.mu, cov=cov))
            except np.linalg.LinAlgError:
                ## Project covariance to positive semi-definite
                if lam is None:
                    lam, V = np.linalg.eigh(self.cov)
                cov = np.linalg.multi_dot( [V, np.diag(np.clip(lam, a_min=tol, a_max=None)), np.transpose(V)] )
                tol = 10*tol

