import abc
import numpy as np
from ball import ProbabilityBall
from kde import KDE
from i_projection import kl_minimization
from tqdm import trange
from copy import deepcopy
from scipy.special import xlogy

'''
    General class of models that can be used with OWL methodology. 
'''
class OWLModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 n:int, ## Number of input points
                 w:np.ndarray=None, ## Weights on the points
                 **kwargs):
        self.n = n
        self.w = w
        if w is None:
            self.w = np.ones(self.n)

    @abc.abstractmethod
    def reinitialize(self, reset_w:bool, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_likelihood(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def maximize_weighted_likelihood(self, **kwargs):
        raise NotImplementedError

    def set_w(self, w:np.ndarray, **kwargs):
        ## w must be length n and non-negative
        non_neg = np.all(w >= 0)
        assert len(w) == self.n, "Attempted to set w with incorrect length vector."
        assert non_neg, "Attempted to set w with negative entries."
        self.w = w

    ## Alternating optimization procedure.
    def fit_owl(self, 
                ball:ProbabilityBall, 
                n_iters:int,
                kde:KDE = None, 
                admmsteps:int=1000, 
                admmtol:float=10e-5, 
                verbose:bool=False, 
                **kwargs):
        
        p = np.zeros(self.n)
        for _ in trange(n_iters, disable=(not verbose)):
            ## Take some EM steps
            self.maximize_weighted_likelihood()

            ## Get likelihood vector of the model
            log_p_theta = self.log_likelihood(**kwargs)

            ## Solve for p
            p = kl_minimization(log_q=log_p_theta, ball=ball, kde=kde, w_init=p, max_iter=admmsteps, eta=0.01, adjust_eta=True, tol=admmtol)
            p = np.clip(p, a_min=0.0, a_max=None)

            ## Normalize to sum to n
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