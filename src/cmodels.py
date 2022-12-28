import abc
import numpy as np
from typing import Union
from balls_kdes import KDE, KDEDensity, ProbabilityBall
from i_projection import kl_minimization
from tqdm import trange
from copy import deepcopy
from scipy.special import xlogy

'''
    General class of "coarse" or "robust" probabalistic models.

    To do maximum likelihood estimation, should implement reinitialize, E_step, and soft_M_step.

    To do OWL estimation, should implement reinitialize, log_likelihood_vector, E_step and hard_M_step.
'''

class CModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 n:int, ## Number of input points
                 hard:bool=True, ## Whether to do hard EM or soft EM 
                 w:np.ndarray=None, ## Weights on the points
                 **kwargs):
        self.n = n
        self.w = w
        self.hard = hard
        if w is None:
            self.w = np.ones(self.n)

    @abc.abstractmethod
    def reinitialize(self, reset_w:bool, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_likelihood_vector(self, **kwargs) -> np.ndarray:
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

    ## HARD EM 
    def EM_step(self, n_steps:int=1, **kwargs):
        for _ in range(n_steps):
            self.E_step()
            if self.hard:
                self.hard_M_step()
            else:
                self.soft_M_step()

    @abc.abstractmethod
    def probability(self, new_X:np.ndarray, **kwargs):
        raise NotImplementedError

    def set_w(self, w:np.ndarray, **kwargs):
        ## w must be length n, non-negative, and sum to n
        non_neg = np.all(w >= 0)
        assert len(w) == self.n, "Attempted to set w with incorrect length vector."
        assert non_neg, "Attempted to set w with negative entries."
        self.w = w

    ## Alternating optimization procedure.
    def am_robust(self, 
                  ball:ProbabilityBall, 
                  n_iters:int,
                  kde:Union[KDE, KDEDensity] = None, 
                  emsteps:int=20, 
                  admmsteps:int=1000, 
                  admmtol:float=10e-5, 
                  verbose:bool=False, 
                  **kwargs):
        p = np.zeros(self.n)
        for _ in trange(n_iters, disable=(not verbose)):
            ## Take some EM steps
            self.EM_step(n_steps=emsteps)

            ## Get likelihood vector of the model
            log_p_theta = self.log_likelihood_vector(**kwargs)

            ## Solve for p
            p = kl_minimization(log_q=log_p_theta, ball=ball, kde=kde, w_init=p, max_iter=admmsteps, eta=0.01, adjust_eta=True, tol=admmtol)
            p = np.clip(p, a_min=0.0, a_max=None)

            ## Normalize to sum to 1
            w = self.n * p/np.sum(p) 

            ## Set w
            self.set_w(w)

'''
    Fits a maximum likelihood model using EM with random restarts.

    model must have implemented reinitialize, E_step, and soft_M_step.
'''

def fit_mle(model:CModel, repeats:int=25):
    best_model = deepcopy(model)
    curr_model = deepcopy(model)
    best_ll = -np.infty
    for _ in range(repeats):
        curr_model.EM_step(n_steps=100)
        ll = np.sum(curr_model.log_likelihood_vector())
        if ll > best_ll:
            best_ll = ll
            best_model = deepcopy(curr_model)

        ## Reinitialize current model
        curr_model.reinitialize(reset_w=True)
    return(best_model)

'''
    Fits an OWL model using alternating optimization with random restarts.

    model must have implemented reinitialize, log_likelihood_vector, E_step and hard_M_step.

    ball must have implemented get_prox_operator.
'''

def fit_owl(model:CModel, ball:ProbabilityBall, repeats=10, admmsteps=1000, verbose:bool=True):
    best_model = deepcopy(model)
    curr_model = deepcopy(model)
    best_ll = -np.infty
    for _ in range(repeats):
        curr_model.am_robust(ball=ball, n_iters=15, kde=None, admmsteps=admmsteps, verbose=verbose)
        prob = curr_model.w/np.sum(curr_model.w)
        ll = np.dot(prob, curr_model.log_likelihood_vector()) - np.nansum(xlogy(prob , prob))
        if ll > best_ll:
            best_ll = ll
            best_model = deepcopy(curr_model)

        ## Reinitialize current model
        curr_model.reinitialize(reset_w=True)
    return(best_model)


'''
    Fits a kernelized OWL model using alternating optimization with random restarts 
    + bandwidth search for the kernel bandwidth.
'''

def fit_kernelized_owl(model:CModel, ball:ProbabilityBall, kde:KDE, bandwidth_schedule:list, repeats=10, admmsteps=1000, verbose:bool=True):
    best_model = deepcopy(model)
    curr_model = deepcopy(model)
    best_ll = -np.infty
    for bandwidth in bandwidth_schedule:
        kde.reset_bandwidth(bandwidth)
        for _ in range(repeats):
                curr_model.am_robust(ball=ball, n_iters=15, kde=kde, admmsteps=admmsteps, verbose=verbose)
                prob = curr_model.w/np.sum(curr_model.w)
                ll = np.dot(prob, curr_model.log_likelihood_vector()) - np.nansum(xlogy(prob , prob))
                if ll > best_ll:
                    best_ll = ll
                    best_model = deepcopy(curr_model)

                ## Reinitialize current model
                curr_model.reinitialize(reset_w=True)
    return(best_model)