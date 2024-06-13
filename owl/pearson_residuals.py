import numpy as np
from owl.models import OWLModel
from owl.kde import RBFKDE
from tqdm import trange, tqdm
from copy import deepcopy


def hellinger_weight_function(delta:np.ndarray):
    a_delta = 2*(np.sqrt(delta + 1)-1)
    num = np.clip(a_delta+1, a_min=0.0, a_max=None )
    denom = 1+delta
    w = np.clip(np.where(delta>-1, num/denom, 1), a_min=0., a_max=1.)
    return(w)

def chisquare_weight_function(delta:np.ndarray):
    w = 1 - np.square(delta)/np.square(delta + 2)
    return(w)


def pearson_residual_continuous(model:OWLModel, 
                                  kde:RBFKDE, 
                                  n_iters:int=20,
                                  weight_fn:str='hellinger',
                                  verbose:bool=False):
    
    '''
        Assume that the vectors model.log_likelihood() and kde.log_likelihood() 
        correspond to the same ordering of the data.
    '''
    kde_log_likelihood = kde.log_likelihood() 
    weight_function = hellinger_weight_function if weight_fn=='hellinger' else chisquare_weight_function
    for _ in trange(n_iters, disable=(not verbose)):
        model.maximize_weighted_likelihood()

        model_log_likelihood = model.rbf_kernel_smoothed_log_likelihood(h=kde.bandwidth)
        ## Clip at extremal values without going to infinity
        log_likelihood_diff = np.clip(kde_log_likelihood - model_log_likelihood, a_min=-700, a_max=700)
        delta = np.exp(log_likelihood_diff) - 1 ## Pearson residual

        w = weight_function(delta)
        model.set_w(w)
    
    model.maximize_weighted_likelihood()

    return(model)


def pro_hellinger_selection(model:OWLModel, kde_list:list[RBFKDE], n_iters:int=20, verbose:bool=False):
    best_model = model
    best_distance = np.inf
    best_kde = None
    
    for kde in tqdm(kde_list, disable=(not verbose)):
        curr_model = deepcopy(model)
        curr_model = pearson_residual_continuous(curr_model, kde=kde, n_iters=n_iters, weight_fn='hellinger', verbose=False)

        model_log_likelihood = model.log_likelihood()
        kde_log_likelihood = kde.log_likelihood()

        hellinger_distance = 0.5*np.mean(np.square(1 - np.sqrt(np.exp(model_log_likelihood - kde_log_likelihood))))
        if hellinger_distance < best_distance:
            best_distance = hellinger_distance
            best_model = curr_model
            best_kde = kde

    return(best_model, best_distance, best_kde)

def pearson_residual_discrete(model:OWLModel, 
                              n_iters:int=20, 
                              weight_fn:str='hellinger',
                              verbose:bool=False):
    
    idx_unique = model.unique_mapping()
    idxs, inverse, counts = np.unique(idx_unique, return_inverse=True, return_counts=True)
    empirical_log_prob = np.log(counts/np.sum(counts))
    weight_function = hellinger_weight_function if weight_fn=='hellinger' else chisquare_weight_function
    
    for _ in trange(n_iters, disable=(not verbose)):
        model.maximize_weighted_likelihood()

        ## Convert likelihoods of individual observations to likelihoods of discrete elements
        model_log_likelihood = model.log_likelihood()
        model_log_likelihood = model_log_likelihood[idxs]
        
        ## Clip at extremal values without going to infinity
        log_likelihood_diff = np.clip(empirical_log_prob - model_log_likelihood, a_min=-700, a_max=700)
        delta = np.exp(log_likelihood_diff) - 1 ## Pearson residual

        w = weight_function(delta)

        ## Remap weights to individual data points
        w = w/counts
        w = w[inverse]

        model.set_w(w)
    
    model.maximize_weighted_likelihood()

    return(model)