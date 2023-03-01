import abc
import numpy as np
from owl.models import OWLModel
from owl.ball import ProbabilityBall
from owl.kde import KDE
from owl.kde import knn_bandwidth
from scipy.spatial.distance import cdist, pdist
from copy import deepcopy
from sklearn.cluster import AgglomerativeClustering
from scipy.special import xlogy, logsumexp
from scipy.optimize import linear_sum_assignment

'''
    Generic models that can be fit using hard or soft EM. Typically, mixture models.
'''
class OWLMixtureModel(OWLModel):
    def __init__(self,
                 n:int, ## Number of input points
                 w:np.ndarray=None, ## Weights on the points
                 hard:bool=True, ## Whether or not maximization is done via hard or soft EM
                 em_steps:int=20, ## How many EM steps to do for weighted likelihood maximization
                 repeats:int=10, ## How many random repeats will we perform when fitting MLE or OWL?
                 **kwargs):
        super().__init__(n=n,w=w, **kwargs)
        self.hard = hard
        self.em_steps = em_steps
        self.repeats = repeats

    @abc.abstractmethod
    def reinitialize(self, reset_w:bool, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def E_step(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def hard_M_step(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def soft_M_step(self, **kwargs):
        raise NotImplementedError

    ## Weighted likelihood maximization happens via EM.
    def maximize_weighted_likelihood(self, **kwargs):
        for _ in range(self.em_steps):
            self.E_step()
            if self.hard:
                self.hard_M_step()
            else:
                self.soft_M_step()


    def owl_step(self, ball: ProbabilityBall, n_iters:int, kde: KDE, admmsteps:int, admmtol:float, eta:int, verbose:bool):
        super().fit_owl(ball=ball, n_iters=n_iters, kde=kde, admmsteps=admmsteps, admmtol=admmtol, eta=eta, verbose=verbose)

    def fit_owl(self, 
                ball: ProbabilityBall, 
                n_iters: int=15, 
                kde: KDE = None, 
                bandwidth_schedule:list = None,
                admmsteps: int = 1000, 
                admmtol: float = 0.0001,
                eta: float = 0.01,
                verbose: bool = False, 
                **kwargs):
        
        best_dict = deepcopy(self.__dict__)
        best_ll = -np.infty
        if (bandwidth_schedule is None) and (kde is not None):
            bandwidth_schedule = [kde.bandwidth]
        elif bandwidth_schedule is None:
            bandwidth_schedule = [0]

        for bw in bandwidth_schedule:
            if (kde is not None) and (kde.bandwidth != bw):
                kde.recalculate_kernel(bandwidth=bw)
            
            for _ in range(self.repeats):
                self.owl_step(ball=ball, n_iters=n_iters, kde=kde, admmsteps=admmsteps, admmtol=admmtol, eta=eta, verbose=verbose)
                prob = self.w/np.sum(self.w)
                ll = np.dot(prob, self.log_likelihood()) - np.nansum(xlogy(prob , prob))
                if ll > best_ll:
                    best_ll = ll
                    best_dict = deepcopy(self.__dict__)

                ## Reinitialize current model
                self.reinitialize(reset_w=True)
        
        ## Update model to be the best found model
        self.__dict__.update(best_dict)

    def fit_mle(self, **kwargs):
        best_dict = deepcopy(self.__dict__)
        best_ll = -np.infty
        for _ in range(self.repeats):
            self.maximize_weighted_likelihood()
            ll = np.sum(self.log_likelihood())
            if ll > best_ll:
                best_ll = ll
                best_dict = deepcopy(self.__dict__)

        ## Update model to be the best found model
        self.__dict__.update(best_dict)

'''
    Mixture of spherical Gaussians.
'''
class SphericalGMM(OWLMixtureModel):
    def __init__(self, 
                X: np.ndarray, ## Input samples [n samples x p features]
                K:int, ## Number of components
                w:np.ndarray = None, 
                hard:bool = True, 
                em_steps:int=20,
                repeats:int=10, ## How many random repeats will we perform when fitting MLE or OWL?
                **kwargs
                ):
        self.X = deepcopy(X)
        n, self.p = X.shape
        super().__init__(n=n, w=w, hard=hard, em_steps=em_steps, repeats=repeats, **kwargs)
        self.K = K

        ## Need a lower bound on variances
        self.var_lower_bound = 0.1*np.min(pdist(self.X, metric='sqeuclidean')) ## Minimum distance between two points.
        self.solo_var = np.square(knn_bandwidth(self.X, k=5))


        ## Allocate parameters
        self.cluster_sizes = np.zeros(K)
        self.cluster_weights = np.zeros(K)
        self.mu = np.zeros((K, self.p))
        self.tau = np.zeros(K)

        ## Initialize clusters
        self.reinitialize(reset_w=False)

    def reinitialize(self, reset_w: bool, **kwargs) -> None:
        if reset_w:
            self.reset_w()

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
        with np.errstate(divide = 'ignore'):
            log_probs = np.log(self.pi) + 0.5*self.p*np.log(self.tau) - 0.5*self.tau*square_dists
        probs = np.exp( log_probs - np.max(log_probs, axis=1, keepdims=True) )
        self.probs = probs/np.sum(probs, axis=1, keepdims=True)

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

    def log_likelihood(self):
        if self.hard:
            with np.errstate(divide = 'ignore'):
                log_prior_probs = np.log(self.pi[self.z])
            taus = self.tau[self.z]
            return(log_prior_probs + 0.5*self.p*np.log(taus) - 0.5*taus*np.sum(np.square(self.X - self.mu[self.z]), axis=1))
        else:
            square_dists = cdist(self.X, self.mu, metric='sqeuclidean') ## n x K
            with np.errstate(divide = 'ignore'):
                expanded_result = np.log(self.pi) + 0.5*self.p*np.log(self.tau) - 0.5*self.tau*square_dists
            return(logsumexp(expanded_result, axis=1))

    def mean_mse(self, mu:np.ndarray): 
        dist_mat = cdist(self.mu, mu, metric='sqeuclidean')
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        return(dist_mat[row_ind, col_ind].mean())




class GeneralGMM(OWLMixtureModel):
    def __init__(self, 
        X: np.ndarray, ## Input samples [n samples x p features]
        K:int, ## Number of components
        w:np.ndarray = None,
        hard:bool = True, 
        em_steps:int=20,
        repeats:int=10, ## How many random repeats will we perform when fitting MLE or OWL?
        **kwargs
        ):
        self.X = deepcopy(X)
        n, self.p = X.shape
        super().__init__(n=n, w=w, hard=hard, em_steps=em_steps, repeats=repeats, **kwargs)
        self.K = K

        self.var_lower_bound = 0.1*np.min(pdist(self.X, metric='sqeuclidean')) ## Minimum distance between two points.
        self.solo_var = np.square(knn_bandwidth(self.X, k=5))


        self.cluster_sizes = np.zeros(K)
        self.cluster_weights = np.zeros(K)
        self.mu = np.zeros((K, self.p))
        self.prec_mats = np.zeros((K, self.p, self.p))

        ## Initialize clusters
        self.reinitialize(reset_w=False)

    def reinitialize(self, reset_w: bool, **kwargs) -> None:
        if reset_w:
            self.reset_w()

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
        with np.errstate(divide = 'ignore'):
            log_probs = np.log(self.pi)+ 0.5*log_dets - 0.5*D
        probs = np.exp( log_probs - np.max(log_probs, axis=1, keepdims=True))
        self.probs = probs/np.sum(probs, axis=1, keepdims=True)

        ## Hard EM -- assignment
        self.z = np.argmax(log_probs, axis=1)

    def log_likelihood(self):
        if self.hard:
            with np.errstate(divide = 'ignore'):
                log_prior_probs = np.log(self.pi[self.z])

            diff = self.X - self.mu[self.z]
            sq_mahal = np.einsum('ij, ik, ijk -> i' , diff, diff, self.prec_mats[self.z])
            _, log_dets = np.linalg.slogdet(self.prec_mats)

            return(log_prior_probs + 0.5*log_dets[self.z] - 0.5*sq_mahal)
        else:
            with np.errstate(divide = 'ignore'):
                log_prior_probs = np.log(self.pi)
            _, log_dets = np.linalg.slogdet(self.prec_mats)
            log_mat = np.zeros((self.n, self.K))
            for k in range(self.K):
                log_mat[:,k] = log_prior_probs[k] + 0.5*log_dets[k] - 0.5*(cdist(self.X, (self.mu[k,:])[np.newaxis,:], metric='mahalanobis', VI=self.prec_mats[k])).squeeze()
            return(logsumexp(log_mat, axis=1))



'''
    Implements a product Bernoulli mixture model.

    Data points are stored in the rows of X.
'''
class BernoulliMM(OWLMixtureModel):
    def __init__(self, 
        X: np.ndarray, ## Input samples [2d array]
        K:int, ## Number of components
        alpha:float = 0.5, ## Beta prior parameter
        beta:float = 0.5, ## Beta prior parameter
        w:np.ndarray = None, ## Weights over the samples (set to None for uniform)
        hard:bool=True, ## Whether or not maximization is done via hard or soft EM
        em_steps:int=20, ## How many EM steps to do for weighted likelihood maximization
        repeats:int=10, ## How many repeats
        **kwargs
        ):
        self.X = deepcopy(X)
        n, self.p = X.shape
        super().__init__(n=n, w=w, hard=hard, em_steps=em_steps, repeats=repeats, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.K = K

        self.wX = np.einsum('ij, i -> ij', self.X, self.w) 

        ## Latent variables
        self.reinitialize(reset_w=False)


    def reinitialize(self, reset_w: bool, **kwargs) -> None:
        self.z = np.random.choice(self.K, size=self.n, replace=True).astype(int)
        self.pi = np.random.dirichlet(alpha=(1.0/self.K * np.ones(self.K)))
        self.lam = np.random.beta(a=self.alpha, b=self.beta, size=(self.K, self.p))
        if reset_w:
            self.reset_w()
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
        probs = probs*self.w[:,np.newaxis]

        self.pi = np.mean(probs, axis=0)
        self.lam = np.dot( np.transpose(probs), self.X )/(np.sum(probs, axis=0))[:,np.newaxis]
        

    def log_likelihood(self):
        if self.hard:
            with np.errstate(divide = 'ignore'):
                log_prior_probs = np.log(self.pi[self.z])
                log_pos_probs = np.sum(xlogy(self.X, np.clip(self.lam[self.z],  a_min=10e-20, a_max=None) ), axis=1)
                log_neg_probs = np.sum(xlogy(1. - self.X, np.clip(1. - self.lam[self.z], a_min=10e-20, a_max=None)), axis=1)

            return(log_prior_probs + log_pos_probs + log_neg_probs)
        else:
            log_mat = np.zeros((self.n, self.K))
            with np.errstate(divide = 'ignore'):
                log_prior_probs = np.log(self.pi)
                pos_lam =  np.clip(self.lam,  a_min=10e-20, a_max=None) ## K x p
                neg_lam = np.clip(1. - self.lam, a_min=10e-20, a_max=None) ## K x p
                for k in range(self.K):
                    log_mat[:,k] = log_prior_probs[k] +  np.sum(xlogy(self.X, pos_lam[k,:]), axis=1) + np.sum(xlogy(1. - self.X, neg_lam[k,:]), axis=1)

            return(logsumexp(log_mat, axis=1))

    def cooccurrence(self) -> np.ndarray:
        C_tens = np.einsum('hi, hj -> hij', self.lam, self.lam)
        C = np.einsum('hij, h -> ij', C_tens, self.pi)
        return(C)

    def mean_mae(self, lam): ## Mean absolute error
        mean_l1 = lambda x,y: np.mean(np.abs(x - y))
        dist_mat = cdist(self.lam, lam, metric=mean_l1)
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        return(dist_mat[row_ind, col_ind].mean())