import numpy as np
from cmodels import CModel
from scipy.special import xlogy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def pred_cooccurrence(lam: np.ndarray, pi: np.ndarray):
    C_tens = np.einsum('hi, hj -> hij', lam, lam)
    C = np.einsum('hij, h -> ij', C_tens, pi)
    return(C)


class BernoulliMM(CModel):
    def __init__(self, 
        X: np.ndarray, ## Input samples [2d array]
        K:int, ## Number of components
        alpha:float = 0.5, ## Beta prior parameter
        beta:float = 0.5, ## Beta prior parameter
        w:np.ndarray = None ## Weights over the samples (set to None for uniform)
        ):
        self.X = X
        n, self.p = X.shape
        super().__init__(n=n, w=w)

        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.dir_alpha = 1.0/K * np.ones(K)

        self.wX = np.einsum('ij, i -> ij', self.X, self.w) 

        ## Latent variables
        self.z = np.random.choice(self.K, size=self.n, replace=True).astype(int)
        self.pi = np.random.dirichlet(alpha=self.dir_alpha)
        self.lam = np.random.beta(a=self.alpha, b=self.beta, size=(self.K, self.p))


    def reinitialize(self, reset_w: bool, **kwargs) -> None:
        self.z = np.random.choice(self.K, size=self.n, replace=True).astype(int)
        self.pi = np.random.dirichlet(alpha=self.dir_alpha)
        self.lam = np.random.beta(a=self.alpha, b=self.beta, size=(self.K, self.p))
        if reset_w:
            self.w = np.ones(self.n)
            self.wX = np.einsum('ij, i -> ij', self.X, self.w) 

    def set_w(self, w: np.ndarray, **kwargs):
        super().set_w(w, **kwargs)
        self.wX = np.einsum('ij, i -> ij', self.X, self.w)

    def E_step(self):
            lam = np.clip(self.lam, a_min=1e-10, a_max=None) ## Numerical stability
            one_m_lam = np.clip(1.0-self.lam, a_min=1e-10, a_max=None) ## Numerical stability

            log_pos_probs = np.dot(self.X, np.transpose(np.log(lam) ))
            log_neg_probs =  np.dot( (1.- self.X), np.transpose(np.log(one_m_lam)) )


            with np.errstate(divide = 'ignore'):
                log_pi = np.log(self.pi)
            
            self.log_probs = log_pi + log_neg_probs + log_pos_probs
            self.z = np.argmax(self.log_probs, axis=1)

    def hard_M_step(self, **kwargs):
        self.lam = 0. * self.lam
        self.pi = 0. * self.pi
        for k in range(self.K):
            cluster_mask = self.z == k
            cluster_weight = np.sum(self.w[cluster_mask])
            if cluster_weight > 0.0:
                self.lam[k] = np.sum(self.wX[cluster_mask], axis=0)/cluster_weight
                self.pi[k] = np.sum(cluster_mask)/self.n

    def soft_M_step(self, **kwargs):
        probs = np.exp(self.log_probs - np.max(self.log_probs, axis=1, keepdims=True))
        probs = probs/np.sum(probs, axis=1, keepdims=True)

        self.pi = np.mean(probs, axis=0)
        self.lam = np.dot( np.transpose(probs), self.X )/(np.sum(probs, axis=0))[:,np.newaxis]
        

    def log_likelihood_vector(self):
        with np.errstate(divide = 'ignore'):
            log_prior_probs = np.log(self.pi[self.z])
            log_pos_probs = np.sum(xlogy(self.X, np.clip(self.lam[self.z],  a_min=10e-20, a_max=None) ), axis=1)
            log_neg_probs = np.sum(xlogy(1.- self.X, np.clip(1. - self.lam[self.z], a_min=10e-20, a_max=None)), axis=1)

        ll_vec = log_prior_probs + log_pos_probs + log_neg_probs

        return(ll_vec)

    def cooccurrence(self) -> np.ndarray:
        C_tens = np.einsum('hi, hj -> hij', self.lam, self.lam)
        C = np.einsum('hij, h -> ij', C_tens, self.pi)
        return(C)

    def cooccurrence_distance(self, C:np.ndarray):
        cooccur_dist = np.mean(np.abs(C - self.cooccurrence()))
        return(cooccur_dist)

    def mean_mae(self, lam): ## Mean absolute error
        mean_l1 = lambda x,y: np.mean(np.abs(x - y))
        dist_mat = cdist(self.lam, lam, metric=mean_l1)
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        return(dist_mat[row_ind, col_ind].mean())

    def adjusted_rand_index(self, z:np.ndarray, mask:np.ndarray=None): ## Adjusted rand index
        if mask is None:
            ari = adjusted_rand_score(z, self.z)
        else:
            ari = adjusted_rand_score(z[mask], self.z[mask])
        return(ari)