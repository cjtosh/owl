import numpy as np
from cmodels import CModel
from balls_kdes import knn_bandwidth
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import AgglomerativeClustering
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

class SphericalGMM(CModel):
    def __init__(self, 
        X: np.ndarray, ## Input samples [n samples x p features]
        K:int = 10, ## Number of components
        hard:bool = True, ## Hard EM or soft EM
        w:np.ndarray = None, ## Weights over the samples (set to None for uniform)
        ):
        self.X = deepcopy(X)
        n, self.p = X.shape
        super().__init__(n=n, w=w, hard=hard)

        self.K = K

        ## Need a lower bound on variances
        self.var_lower_bound = 0.1*np.min(pdist(self.X, metric='sqeuclidean')) ## Minimum distance between two points.
        self.solo_var = np.square(knn_bandwidth(self.X, k=5))

        ## Initialize via hierarchical clustering
        clus = AgglomerativeClustering(n_clusters=self.K, linkage='ward')
        clus.fit(self.X)
        self.z = deepcopy(clus.labels_)

        self.cluster_sizes = np.zeros(K)
        self.cluster_weights = np.zeros(K)
        self.mu = np.zeros((K, self.p))
        self.tau = np.zeros(K)

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
    
    def E_step(self):
        square_dists = cdist(self.X, self.mu, metric='sqeuclidean')
        
        log_probs = np.log(self.pi) + 0.5*self.p*np.log(self.tau) - 0.5*self.tau*square_dists
        probs = np.exp( log_probs - np.max(log_probs, axis=1, keepdims=True) )
        self.probs = probs/np.sum(probs, axis=1, keepdims=True)
        self.probs = self.probs * self.w[:,np.newaxis] ## Reweight the probabilities

        self.z = np.argmax(log_probs, axis=1)

    def hard_M_step(self):
        for h in range(self.K):
            mask = self.z==h
            self.cluster_sizes[h] = np.sum(mask)
            self.cluster_weights[h] = np.sum(self.w[mask])
            if (self.cluster_weights[h] > 0) and (self.cluster_sizes[h] > 1):
                self.mu[h,:] = np.average(self.X[mask,:], axis=0, weights=self.w[mask])

                sq_dists = np.sum(np.square( self.X[mask,:] - self.mu[h,:]), axis=1)
                variance = np.average(sq_dists, weights=self.w[mask])
                variance = np.clip(variance, a_min=self.var_lower_bound, a_max=None)

                self.tau[h] = (self.p * self.cluster_sizes[h])/variance
            elif (self.cluster_sizes[h] == 1):
                self.mu[h,:] = np.squeeze(self.X[mask,:])
                variance = self.solo_var
                self.tau[h] = self.p/variance
            else:
                idx = np.random.choice(self.n)
                self.mu[h,:] = self.X[idx, :]
                variance = self.solo_var
                self.tau[h] = self.p/variance
        
        self.pi = self.cluster_sizes/self.n

    def soft_M_step(self):
        self.pi = np.mean(self.probs, axis=0)
        self.pi = self.pi/np.sum(self.pi)

        for h in range(self.K):
            if self.pi[h] > 0:
                self.mu[h, :] = np.average(self.X, axis=0, weights=self.probs[:,h])
                sq_dists = np.sum(np.square( self.X - self.mu[h,:]), axis=1)
                variance = np.average(sq_dists, weights=self.probs[:,h])
                variance = np.clip(variance, a_min=self.var_lower_bound, a_max=None)
                self.tau[h] = (self.p * self.n * self.pi[h])/variance
            else:
                idx = np.random.choice(self.n)
                self.mu[h,:] = self.X[idx, :]
                variance = self.solo_var
                self.tau[h] = self.p/variance

    def log_likelihood_vector(self):
        log_prior_probs = np.log(self.pi[self.z])

        taus = self.tau[self.z]
        return(log_prior_probs + 0.5*self.p*np.log(taus) - 0.5*taus*np.sum(np.square(self.X - self.mu[self.z]), axis=1))

    def probability(self, new_X:np.ndarray):
        square_dists = cdist(new_X, self.mu, metric='sqeuclidean')
        scaled_square_dists = np.einsum('ih, h -> ih', square_dists, self.tau)

        result = np.einsum('ih, h, h -> i', np.exp(-0.5*scaled_square_dists), np.power(self.tau, 0.5*self.p), self.pi)
        return(result)

    def hellinger_distance(self, mu, tau):
        sigma_1_sq = 1./tau
        sigma_2_sq = 1./self.tau
        sigma_1 = np.sqrt(sigma_1_sq)
        sigma_2 = np.sqrt(sigma_2_sq)
        C = np.empty((self.K,self.K))
        for i in range(self.K):
            for j in range(self.K):
                dist = np.sum(np.square(self.mu[i] - mu[j]))
                sum_sq = sigma_1_sq[i] + sigma_2_sq[j]
                C[i, j] = 1. - np.power(2.0*sigma_1[i]*sigma_2[j]/sum_sq, self.p/2.) * np.exp(-0.25*dist/sum_sq)

        C = np.sqrt(C)
        row_ind, col_ind = linear_sum_assignment(C)
        return(C[row_ind, col_ind].mean())

    def mean_mse(self, mu:np.ndarray): 
        dist_mat = cdist(self.mu, mu, metric='sqeuclidean')
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        return(dist_mat[row_ind, col_ind].mean())

    def adjusted_rand_index(self, z:np.ndarray, mask:np.ndarray=None): ## Adjusted rand index
        if mask is None:
            ari = adjusted_rand_score(z, self.z)
        else:
            ari = adjusted_rand_score(z[mask], self.z[mask])
        return(ari)