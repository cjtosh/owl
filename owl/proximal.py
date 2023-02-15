import abc
import numpy as np
from numpy.linalg import norm, multi_dot
import scipy.special


class ProximalOperator(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, tilt:bool = False, **kwargs):
        super().__init__()
        self.tilt = tilt

    def __call__(self, z: np.ndarray, y: np.ndarray, eta:float):
        raise NotImplementedError

    def is_tilted(self):
        return(self.tilt)

## Euclidean projection of a vector x to the probability simplex
def proj_simplex(x: np.ndarray):
    n = len(x)
    u = -np.sort(-x)
    avg_csm1 = (np.cumsum(u) - 1.0)/np.arange(1, n+1)
    thresh = u > avg_csm1
    ## Due to large numbers, we may have thresh[0] = False.
    thresh[0] = True
    idx = np.where(thresh)[0][-1]
    lam = avg_csm1[idx]
    return(np.maximum(x - lam, 0.0))

class ProxSimplex(ProximalOperator):
    def __init__(self, tilt:bool = False, **kwargs):
        super().__init__(tilt=tilt)
    
    def __call__(self, z: np.ndarray, y: np.ndarray, eta:float):
        return(proj_simplex(x=(z + y*eta)))

## Euclidean projection of a vector x to unit norm l1 ball
def project_unit_l1_ball(x: np.ndarray):
    l1_norm = np.sum(np.abs(x))
    if l1_norm <= 1.0:
        return(x)
    else:
        z = proj_simplex(np.abs(x))
        return(z*np.sign(x))

## Euclidean projection of a vector x ball {v : |x - v|_1 <= r}
def proj_l1_ball(x: np.ndarray, c: np.ndarray, r: float = 1.0):
    result = c + r*project_unit_l1_ball((x-c)/r)
    return(result)


class ProxL1Ball(ProximalOperator):
    def __init__(self, c:np.ndarray, r:float, tilt:bool = False, **kwargs):
        super().__init__(tilt=tilt)
        self.center = c
        self.r = r

    def __call__(self, z: np.ndarray, y: np.ndarray, eta:float):
        result = proj_l1_ball(x=(z + y*eta), c=self.center, r=self.r)
        return(result)

## Euclidean projection of a vector x ball {v : |x - v|_2 <= r}
def proj_l2_ball(x: np.ndarray, c: np.ndarray, r: float = 1.0):
    diff = x - c
    l2_dist = np.sqrt(np.sum(np.square(diff)))
    if l2_dist <= r:
        result = x
    else:
        result = c + diff*(r/l2_dist)
    return(result)

class ProxL2Ball(ProximalOperator):
    def __init__(self, c:np.ndarray, r:float, tilt:bool = False, **kwargs):
        super().__init__(tilt=tilt)
        self.center = c
        self.r = r

    def __call__(self, z: np.ndarray, y: np.ndarray, eta:float):
        result = proj_l2_ball(x=(z + y*eta), c=self.center, r=self.r)
        return(result)

## Log scale of Lambert's W function (this is apparently Wright's omega)
def lambertw_logscale(x: np.ndarray):
    return(np.real(scipy.special.wrightomega(x)))

## Proximal operator for p -> KL(p | q)
def KL_prox(p: np.ndarray, q: np.ndarray, lam: float):
    result = lam * lambertw_logscale( np.log(q) + (p/lam - 1.0) - np.log(lam))
    return(result)

def KL_prox_log_q(p: np.ndarray, log_q: np.ndarray, lam: float):
    result = lam * lambertw_logscale( log_q + (p/lam - 1.0) - np.log(lam))
    return(result)

## Proximal operator for p -> sum_i p_i log p_i
def entropy_prox(p: np.ndarray, lam: float):
    result = lam * lambertw_logscale( (p/lam - 1.0) - np.log(lam))
    return(result)


class ProxKLLogScale(ProximalOperator):
    def __init__(self, log_q:np.ndarray, normalize:bool=True, tilt:bool = False, **kwargs):
        super().__init__(tilt=tilt)
        self.log_q = log_q
        if normalize:
            self.log_q  = log_q - scipy.special.logsumexp(log_q)

    def __call__(self, z: np.ndarray, y: np.ndarray, eta:float):
        result = entropy_prox(p=(z + y*eta + eta*self.log_q), lam=eta)
        return(result)

## Proximal operator for function x -> log( sum_i  exp(x_i + log_q_i) )
def lse_prox(x:np.ndarray, log_q:np.ndarray, lam:float):
    result = x - lam * KL_prox_log_q(p=(x/lam), log_q=log_q, lam=(1.0/lam))
    return(result)

## Proximal operator for function x -> |x|_inf
def linf_prox(x: np.ndarray, lam: float):
    z = project_unit_l1_ball(x/lam)
    return(x - lam*z)


## Proximal operator for function x -> epsilon*|x|_inf - <x,c>
def linf_dot_prox(x:np.ndarray, c:np.ndarray, epsilon:float, lam:float):
    result = linf_prox(x=(x + lam*c), lam=(lam*epsilon))
    return(result)


## Finds x that minimizes 0.5*|x - c|^2 such that x is in the ellipsoid described by mu, A, r
## I.e. need x to satisfy (x - mu)^T A (x - mu) < r
## A is specified via its eigendecomposition: Q diag(L) Q^T
## Solved via Halley's method.
def proj_ellipsoid(c: np.ndarray, mu: np.ndarray, Q:np.ndarray, L:np.ndarray, T:int, r:float, xinit:float=None, rtol:float=10e-3, verbose:bool=False):
    c_shift = c - mu
    L_scale = L/r

    dist = multi_dot( [c_shift.T, Q, np.diag(L_scale), Q.T, c_shift])
    if dist <= 1.0:
        return(c, None)
    
    v = np.square(np.dot(Q.T, c_shift))
    if xinit is None:
        x = 1.0/dist
    else:
        x = xinit

    for t in range(T):
        l_n = 1.0 + x*L_scale
        g_n = np.dot(L_scale/np.power(l_n, 2), v) - 1.0 ## G - value
        gp_n = -2.0*np.dot( np.power(L_scale, 2)/np.power(l_n, 3), v) ##G' - value
        gpp_n = 6.0*np.dot( np.power(L_scale, 3)/np.power(l_n, 4), v) ##G'' - value
        x_n = x - (2.0 * g_n * gp_n)/(2.0*(gp_n)**2 - g_n*gpp_n) ## Halley's update
        diff = np.abs(x_n - x)
        x = x_n
        if diff < rtol*np.abs(x_n):
            break
    
    if verbose:
        print(t, "iterations to convergence")
    ## Final solution
    result = multi_dot([Q, np.diag(1.0/(1.0 + x*L_scale)), Q.T, c_shift]) + mu
    return(result, x)


class ProxMMDBall(ProximalOperator):
    def __init__(self, center:np.ndarray, eigL:np.ndarray, Q:np.ndarray, r:float, tilt:bool=False, **kwargs):
        super().__init__(tilt=tilt)
        self.center = center
        self.eigL = eigL
        self.Q = Q
        self.r = r
        self.xinit = None

    def __call__(self, z: np.ndarray, y: np.ndarray, eta:float):
        result, x = proj_ellipsoid(c=(z + y*eta), mu=self.center, Q=self.Q, L=self.eigL, T=100, r=self.r, xinit=self.xinit)
        self.xinit = x
        return(result)


        