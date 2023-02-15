import numpy as np
from numpy.linalg import norm, multi_dot
from owl.ball import ProbabilityBall
from owl.kde import KDE
from owl.proximal import ProximalOperator, ProxSimplex, ProxKLLogScale
from copy import deepcopy


'''
    Finds w that minimizes (kernel adjusted) KL objective.
'''
def kl_minimization(log_q:np.ndarray,       ## log of q probabilities
                    ball:ProbabilityBall,   ## ProbabilityBall object encoding projection operator
                    kde:KDE=None,           ## KDE object (set to None if we are not doing kernel density adjustment)
                    w_init:np.ndarray=None, ## Initial guess
                    max_iter:int=1000,      ## Maximum number of admm steps to take
                    eta:float=0.01,         ## Initial admm penalty parameter
                    adjust_eta:bool=True,   ## Do we adjust eta on the fly?
                    tol:float=10e-5):       ## Relative tolerance to stop at

    simplex_prox = ProxSimplex()
    if kde is None:
        kl_prox = ProxKLLogScale(log_q=log_q, tilt=False)
        ball_prox = ball.get_prox_operator(tilt=False)
        svd = None
        A = None
        prox_ops = [simplex_prox, kl_prox, ball_prox]
    else:
        rowsums = kde.row_sums()
        U, S, Vt = kde.normalized_svd()
        A = kde.normalized_kernel_matrix()
        shifted_log_q = log_q - np.log(rowsums)

        ## KL term gets shifted + tilted by A
        kl_prox = ProxKLLogScale(log_q=shifted_log_q, tilt=True)
        ball_prox = ball.get_prox_operator(tilt=True)
        svd = (U, S, Vt)

        prox_ops = [simplex_prox, kl_prox, ball_prox]
    
    w = consensus_admm(prox_ops=prox_ops, 
                       dim=len(log_q),
                       eta=eta,
                       max_iter=max_iter,
                       xinit=w_init,
                       A=A,
                       svd=svd,
                       adjust_eta=adjust_eta,
                       tol=tol,
                       num_workers=1)
    if A is not None:
        w = np.dot(A, w)
    
    ## Threshold the very low values
    low_mask = w < 0.2/len(w)
    w[low_mask] = 0.0
    return(w)

'''
    Consensus ADMM algorithm.
'''
def consensus_admm(prox_ops:list[ProximalOperator],
                   dim:int,
                   eta:float, 
                   max_iter:int,
                   xinit:np.ndarray=None,
                   A:np.ndarray=None, 
                   svd:tuple[np.ndarray,np.ndarray,np.ndarray]=None,
                   adjust_eta:bool=True,
                   tol:float=10e-5, 
                   num_workers:int=1):

    ## Number of primal variables we need to keep around
    n = len(prox_ops)

    ## Unpack tilting matrix (if it exists)
    if (svd is None) and (A is not None):
        U, S, Vt = np.linalg.svd(A)
    elif (svd is not None):
        U, S, Vt = svd

    x = np.empty((n,dim)) ## Primal variables
    y = np.zeros((n,dim)) ## dual variables

    etas = eta*np.ones(n) ## proximal operator penalty parameters
    rhos = 1.0/etas ## inverse etas

    ## z is the synchronization variable
    if xinit is None:
        z = np.zeros(dim)
    else:
        z = deepcopy(xinit)

    r = np.empty(n) ## primal residuals
    d = np.empty(n) ## dual residuals

    ## Which proximity operators are tilted (by the matrix A)?
    tilted_mask = np.array([prox_op.is_tilted() for prox_op in prox_ops])
    any_tilted = np.any(tilted_mask)
    if any_tilted:
        Az = np.dot(A, z)
    
    for _ in range(max_iter):
        ## First update the primal variables
        x = np.array([prox_ops[i](Az, y[i], etas[i]) if tilted_mask[i] else prox_ops[i](z, y[i], etas[i]) for i in range(n)])

        z_old = deepcopy(z)

        ## Second, update consensus variable
        if any_tilted:
            rhs = np.dot(A, np.sum((rhos[tilted_mask, np.newaxis]*x[tilted_mask] - y[tilted_mask]) , axis=0)) + np.sum((rhos[~tilted_mask, np.newaxis]*x[~tilted_mask] - y[~tilted_mask]) , axis=0)
            inv_diag = np.sum(rhos[tilted_mask])*np.square(S) + np.sum(rhos[~tilted_mask])
            z = multi_dot([Vt.T, np.diag(1.0/inv_diag), Vt, rhs])
            Az = np.dot(A, z)
        else:
            z = (np.einsum('ij, i -> j', x, rhos) - np.sum(y, axis=0))/np.sum(rhos)


        ## Third, update the dual variables
        for i in range(n):
            if prox_ops[i].is_tilted():
                y[i] += (Az - x[i])*rhos[i]
            else:
                y[i] += (z - x[i])*rhos[i]

        ## Fourth, update residuals
        if any_tilted:
            r[tilted_mask] = norm( (x[tilted_mask] - Az) , axis=1)
        r[~tilted_mask] = norm( (x[~tilted_mask] - z) , axis=1)
        d = rhos * norm(z - z_old)
        

        ## Fifth, check for termination
        if tol > 0:
            r_eps = np.sum(np.square(r))
            r_lim = np.maximum( np.sum(np.square(x)), n*np.sum(np.square(z)) )
            d_eps = np.sum(np.square(d))
            d_lim = np.sum(np.square(y))

            if (r_eps < (tol * r_lim)) and (d_eps < (tol * d_lim)):
                break

        ## Sixth, adjust penalty parameters
        if adjust_eta:
            log_ratio = np.log(d) - np.log(r)
            incr_mask = log_ratio > 0.69
            decr_mask = log_ratio < -0.69
            etas[incr_mask] = 1.5 * etas[incr_mask] 
            etas[decr_mask] = 0.75 * etas[decr_mask] 
            rhos = 1.0/etas

    return(z) ## Return the consensus variable