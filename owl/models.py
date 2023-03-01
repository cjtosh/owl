import abc
import numpy as np
from owl.ball import ProbabilityBall
from owl.kde import KDE
from owl.i_projection import kl_minimization
from tqdm import trange
from copy import deepcopy
from joblib import Parallel, delayed
from kneed import KneeLocator
from scipy.signal import savgol_filter
from scipy.special import xlogy

## Parallel run
def prun(jobs: list, nprocs: int):
    return Parallel(n_jobs=nprocs)(jobs)

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
        
        if w is None:
            self.w = np.ones(self.n)
            self.w_init = np.ones(self.n)
        else:
            self.w = w.copy()
            self.w_init = w.copy()
        
    @abc.abstractmethod
    def log_likelihood(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def maximize_weighted_likelihood(self, **kwargs):
        raise NotImplementedError

    def reset_w(self):
        self.w = self.w_init.copy()

    def set_w(self, w:np.ndarray, **kwargs):
        ## w must be length n and non-negative
        non_neg = np.all(w >= 0)
        assert len(w) == self.n, "Attempted to set w with incorrect length vector."
        assert non_neg, "Attempted to set w with negative entries."
        self.w = w.copy()

    ## Alternating optimization procedure.
    def fit_owl(self, 
                ball:ProbabilityBall, 
                n_iters:int=10,
                kde:KDE = None, 
                admmsteps:int=1000, 
                admmtol:float=10e-5,
                eta:float=0.01,
                verbose:bool=False,
                thresh:float=0.2,
                **kwargs):
        
        p = np.zeros(self.n)
        for _ in trange(n_iters, disable=(not verbose)):
            ## Take some EM steps
            self.maximize_weighted_likelihood()

            ## Get likelihood vector of the model
            log_p_theta = self.log_likelihood(**kwargs)

            ## Solve for p
            p = kl_minimization(log_q=log_p_theta, ball=ball, kde=kde, w_init=p, max_iter=admmsteps, eta=eta, adjust_eta=True, thresh=thresh, tol=admmtol)
            p = np.clip(p, a_min=0.0, a_max=None)

            ## Normalize to sum to n
            w = self.n * p/np.sum(p) 

            ## Set w
            self.set_w(w)


## Helper function 
def p_owl_fit(model:OWLModel, 
              ball:ProbabilityBall, 
              n_iters:int=10,
              kde:KDE=None, 
              admmsteps:int=1000, 
              admmtol:float=10e-5,
              eta:float=0.01,
              thresh:float=0.2,
              **kwargs):
    model.fit_owl(ball=ball, n_iters=n_iters, kde=kde, admmsteps=admmsteps, admmtol=admmtol,thresh=thresh, eta=eta)
    prob = model.w/np.sum(model.w)
    val = np.dot(prob, model.log_likelihood()) - np.nansum(xlogy(prob , prob))
    return(model, -val)


def fit_owl(model:OWLModel, 
            ball:ProbabilityBall, 
            n_iters:int=10,
            epsilons:np.ndarray=None, 
            kde:KDE=None, 
            admmsteps:int=1000, 
            admmtol:float=10e-5,
            eta:float=0.01,
            thresh:float=0.2,
            n_workers:int=1,
            percentile:float=90,
            return_all:bool=False,
            **kwargs):

    ## Just one radius
    if epsilons is None:
        epsilons = [ball.r]

    balls = []
    models = []
    for eps in epsilons:
        b = deepcopy(ball)
        b.set_radius(eps)
        balls.append(b)
        m = deepcopy(model)
        models.append(m)

    pfit = delayed(p_owl_fit)
    jobs = (pfit(model=m, ball=b, n_iters=n_iters, kde=kde, admmsteps=admmsteps, admmtol=admmtol, thresh=thresh, eta=eta) for m, b in zip(models, balls))
    result = prun(jobs, n_workers)

    models = []
    values = []
    for m, val in result:
        models.append(m)
        values.append(val)

    if return_all:
        return(models, values)
    elif len(epsilons)<=2:
        idx = np.argmin(values)
        return(models[idx])
    else:
        ## Compute rolling min
        smoothed_values = np.empty(len(values))
        smoothed_values[0] = values[0]
        for i in range(1, len(smoothed_values)):
            smoothed_values[i] = np.min([smoothed_values[i-1], values[i]])

        ## Compute curvature
        deriv_1 = np.gradient(smoothed_values, epsilons)
        deriv_2 = np.gradient(deriv_1, epsilons)
        curv = deriv_2/np.power(1+ np.square(deriv_1), 1.5)

        ## Choose right-most model with largest curvature 
        val = np.percentile(curv, percentile)
        idx = np.max(np.where(curv > val)[0])
        return(models[idx])


