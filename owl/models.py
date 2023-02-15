import abc
import numpy as np
from owl.ball import ProbabilityBall
from owl.kde import KDE
from owl.i_projection import kl_minimization
from tqdm import trange

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