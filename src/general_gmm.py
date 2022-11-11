import numpy as np
from balls_kdes import knn_bandwidth
from cmodels import CModel
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import AgglomerativeClustering
from copy import deepcopy
from sklearn.metrics import adjusted_rand_score



class GeneralGMM(CModel):
    def __init__(self, 
        X: np.ndarray, ## Input samples [n samples x p features]
        K:int = 10, ## Number of components
        w:np.ndarray = None ## Weights over the samples (set to None for uniform)
        ):
        self.X = deepcopy(X)
        n, self.p = X.shape
        super().__init__(n=n, w=w)
        self.K = K
        self.var_lower_bound = 0.1*np.min(pdist(self.X, metric='sqeuclidean')) ## Minimum distance between two points.
        self.solo_var = np.square(knn_bandwidth(self.X, k=5))


        # self.counter = 0
        # self.log_assign_probs = np.zeros(self.n)

        ## Initialize via hierarchical clustering
        clus = AgglomerativeClustering(n_clusters=self.K, linkage='ward')
        clus.fit(self.X)
        self.z = deepcopy(clus.labels_)
        self.cluster_sizes = np.zeros(K)
        self.cluster_weights = np.zeros(K)
        self.mu = np.zeros((K, self.p))
        self.prec_mats = np.zeros((K, self.p, self.p))

        self.hard_M_step()

    def reinitialize(self, reset_w: bool, **kwargs) -> None:
        if reset_w:
            self.w = np.ones(self.n)

        clus = AgglomerativeClustering(n_clusters=self.K, linkage='ward')
        clus.fit(self.X)
        self.z = deepcopy(clus.labels_)

        ## Randomly redistribute a random fraction
        frac = np.random.rand()
        n_change = int(self.n * frac)
        inds = np.random.choice(self.n, size=n_change, replace=False)
        self.z[inds] = np.random.choice(self.K, size=n_change, replace=True)

        ## Take M step
        self.hard_M_step()
        
    def hard_M_step(self, **kwargs):
        for h in range(self.K):
            mask = self.z==h
            self.cluster_sizes[h] = np.sum(mask)
            self.cluster_weights[h] = np.sum(self.w[mask])
            if (self.cluster_weights[h] > 0) and (self.cluster_sizes[h] > 1):
                self.mu[h,:] = np.average(self.X[mask,:], axis=0, weights=self.w[mask])
                cov = np.cov(self.X[mask,:], rowvar=False, ddof=0, aweights=self.w[mask]) + self.var_lower_bound*np.eye(self.p)
            elif (self.cluster_sizes[h] == 1):
                self.mu[h,:] = np.squeeze(self.X[mask,:])
                cov = self.solo_var * np.eye(self.p)
            else:
                idx = np.random.choice(self.n)
                self.mu[h,:] = self.X[idx, :]
                cov = self.solo_var * np.eye(self.p)
            self.prec_mats[h] = np.linalg.inv(cov)

        self.pi = self.cluster_sizes/self.n

    def soft_M_step(self, **kwargs): 
        self.pi = np.mean(self.probs, axis=0)
        self.pi = self.pi/np.sum(self.pi)
        
        for h in range(self.K):
            if self.pi[h] > 0:
                self.mu[h, :] = np.average(self.X, axis=0, weights=self.probs[:,h])
                cov =  np.cov(self.X, rowvar=False, ddof=0, aweights=self.probs[:,h]) + self.var_lower_bound*np.eye(self.p)
            else:
                idx = np.random.choice(self.n)
                self.mu[h,:] = self.X[idx, :]
                cov = self.solo_var * np.eye(self.p)
            self.prec_mats[h] = np.linalg.inv(cov)

    def E_step(self):
        ## D[i,h] = (x_i - mu_h)^T Prec_h (x_i - mu_h) 
        D = np.transpose(np.stack([np.einsum('ij, ik, jk -> i' , (self.X - self.mu[h]), self.X - self.mu[h], self.prec_mats[h])  for h in range(self.K) ]))
        _, log_dets = np.linalg.slogdet(self.prec_mats)
        log_probs = 0.5*log_dets - 0.5*D
        probs = np.exp( log_probs - np.max(log_probs, axis=1, keepdims=True))
        self.probs = probs/np.sum(probs, axis=1, keepdims=True)

        self.probs = self.probs * self.w[:,np.newaxis] ## Reweight the probabilities

        ## Hard EM -- assignment
        self.z = np.argmax(log_probs, axis=1)

        # ## Calculate assignment probabilities
        # self.log_assign_probs = log_probs[np.arange(self.n), self.z]

        # self.counter = (self.counter + 1)%53
        # ## Redistribute very small clusters
        # if self.counter == 0:
        #     for h in range(self.K):
        #         mask = self.z == h
        #         n_size = np.sum(mask)

        #         if (n_size < min(self.K, self.p)) and (n_size > 0):
        #             self.z[mask] = np.random.choice(self.K, size=n_size, replace=True)
        

    def log_likelihood_vector(self):
        log_prior_probs = np.log(self.pi[self.z])

        diff = self.X - self.mu[self.z]
        sq_mahal = np.einsum('ij, ik, ijk -> i' , diff, diff, self.prec_mats[self.z])
        _, log_dets = np.linalg.slogdet(self.prec_mats)

        return(log_prior_probs + 0.5*log_dets[self.z] - 0.5*sq_mahal)

    def probability(self, new_X:np.ndarray):

        ## D[i,h] = (x_i - mu_h)^T Prec_h (x_i - mu_h) 
        D = np.transpose(np.stack([np.einsum('ij, ik, jk -> i' , new_X - self.mu[h], new_X - self.mu[h], self.prec_mats[h])  for h in range(self.K) ]))

        dets = np.linalg.det(self.prec_mats)

        result = np.einsum('ih, h, h -> i', np.exp(-0.5*D), np.power(dets, 0.5), self.pi)
        return(result)

    def adjusted_rand_index(self, z:np.ndarray, mask:np.ndarray=None): ## Adjusted rand index
        if mask is None:
            ari = adjusted_rand_score(z, self.z)
        else:
            ari = adjusted_rand_score(z[mask], self.z[mask])
        return(ari)