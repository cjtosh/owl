import numpy as np
from cmodels import CModel
import scipy.stats as stats
from copy import deepcopy


class Gaussian(CModel):
    def __init__(self, 
        X: np.ndarray, ## Input samples [n samples x p features]
        hard:bool = True, ## Hard EM or soft EM
        w:np.ndarray = None, ## Weights over the samples (set to None for uniform)
        ):
        self.X = deepcopy(X)
        if len(X.shape) == 1:
            self.X = self.X[:,np.newaxis]
        n, self.p = X.shape
        super().__init__(n=n, w=w, hard=hard)

        self.mu = np.zeros(self.p)
        self.cov = np.cov(self.X, rowvar=False, ddof=0)

    def EM_step(self, n_steps:int = 1, **kwargs):
        if n_steps > 0:
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

    def hellinger(self, mu, cov):
        det_1 = np.linalg.det(np.atleast_2d(self.cov))
        det_2 = np.linalg.det(np.atleast_2d(cov))
        comb_det = np.linalg.det(np.atleast_2d( 0.5*cov + 0.5*self.cov))
        comb_inv = np.linalg.inv(np.atleast_2d( 0.5*cov + 0.5*self.cov))

        mu_diff = self.mu - mu
        exp_factor = np.exp(-(1.0/8.0)* np.linalg.multi_dot( [np.transpose(mu_diff), comb_inv, mu_diff] ) )
        sq_hell = 1.0 - ((np.power(det_1, 0.25)*np.power(det_2, 0.25))/np.power(comb_det, 0.5))*exp_factor
        result = np.sqrt(sq_hell)
        return(result)

