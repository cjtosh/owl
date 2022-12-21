import numpy as np
from numpy.linalg import norm, multi_dot
from scipy.spatial.distance import pdist, squareform
from proximal import ProxL1Ball, ProxL2Ball, ProxMMDBall, proj_l1_ball, proj_l2_ball, proj_ellipsoid
from scipy.special import gamma, logsumexp, betainc
from typing import Optional
from sklearn.neighbors import NearestNeighbors, KernelDensity


## Fast bandwidth selection
def knn_bandwidth(X, k):
    nbrs = NearestNeighbors().fit(X)
    ds, _ = nbrs.kneighbors(X, n_neighbors=k)
    return(np.median(ds[:,-1]))

'''
    Class that implements projection onto a probability ball with respect to some distance. 
'''
class ProbabilityBall():
    def __init__(self, 
                 dist_type:str,  ## 'l1', 'l2', or 'mmd'
                 n:int, ## Dimension of the probability vector
                 r:float, ## Radius of the ball
                 points:Optional[np.ndarray]=None, ## OPTIONAL: Points that the probability vector is defined over
                 counts:Optional[np.ndarray]=None, ## OPTIONAL: Counts of the points. If none, then this is np.ones(n)
                 center:Optional[np.ndarray]=None, ## OPTIONAL: Center of the probability vector. If None, then use counts/np.sum(counts)
                 kernel_matrix:Optional[np.ndarray]=None ## OPTIONAL: The kernel matrix of the MMD metric.
                 ):
        self.dist_type = dist_type
        self.points = points
        self.center = center
        self.r = r
        if self.center is None: ## If no center is given, use the empirical measure over points
            if counts is not None:
                self.center = counts/np.sum(counts)
            else:
                self.center = np.ones(n)/float(n)


        if (dist_type == 'mmd') or (kernel_matrix is not None):
            assert (kernel_matrix is not None), "Error -- Cannot use MMD metric without kernel matrix."
            a, b = kernel_matrix.shape
            assert ((a==n) and (b==n)), "Error -- kernel matrix has shape " + str(a) + " x " + str(b) + ", expected shape "  + str(n) + " x " + str(n)
            eigL, self.Q = np.linalg.eigh(kernel_matrix)
            self.eigL = np.clip(eigL, a_min=0.0, a_max=None) ## Clip eigenvalues at 0
            
    def projection(self, x:np.ndarray):
        if self.dist_type == 'l2':
            return(proj_l2_ball(x=x, c=self.center, r=self.r))
        elif self.dist_type == 'l1':
            return(proj_l1_ball(x=x, c=self.center, r=self.r))
        else:
            return(proj_ellipsoid(c=x, mu=self.center, Q=self.Q, L=self.eigL, T=100, r=self.r))

    
    ## A: Tilting matrix (Will project Aw instead of w)
    def get_prox_operator(self, tilt:bool=False):
        if self.dist_type == 'l2':
            return(ProxL2Ball(c=self.center, r=self.r, tilt=tilt))
        elif self.dist_type == 'l1':
            return(ProxL1Ball(c=self.center, r=self.r, tilt=tilt))
        else:
            return(ProxMMDBall(center=self.center, eigL=self.eigL, Q=self.Q, r=self.r, tilt=tilt))


    def check_projection(self, x:np.ndarray, tol:float):
        diff =  x - self.center
        if self.dist_type == 'l2':
            return( norm(diff) < (self.r + tol) )
        elif self.dist_type == 'l1':
            return( norm(diff, ord=1 ) < (self.r + tol) )
        else:
            return( multi_dot([ diff.T, self.Q, np.diag(self.eigL), self.Q.T, diff]) < (self.r + tol) )

    def projection_value(self, x:np.ndarray):
        diff =  x - self.center
        if self.dist_type == 'l2':
            return( norm(diff) )
        elif self.dist_type == 'l1':
            return( norm(diff, ord=1) )
        else:
            return( multi_dot([ diff.T, self.Q, np.diag(self.eigL), self.Q.T, diff]) )


'''
    Lightweight version of KDE class that only keeps track of the densities
'''
class KDEDensity():
    def __init__(self, X: np.ndarray, bandwidth: float, method:str = 'rbf'):
        if method=='rbf':
            method = 'gaussian'
        kde = KernelDensity(bandwidth=bandwidth, kernel=method)
        kde.fit(X=X)
        self.log_likelihood_vals = kde.score_samples(X=X)

    def log_likelihood(self):
        return(self.log_likelihood_vals)


def hat_kernel(distance_matrix:np.ndarray, bandwidth:float, dim:int):
    indicator = (distance_matrix <= bandwidth).astype(float)

    ## Volume of d-dimensional ball of radius bandwidth
    vol_ = np.power(np.pi, 0.5*dim)*np.power(bandwidth, dim)/gamma(0.5*dim + 1.) 
    kernel_mat = indicator/vol_

    reg_inc_beta_args = indicator*(1.0 - 0.25*np.square(distance_matrix/bandwidth))
    mmd_mat = betainc( 0.5*(dim + 1.), 0.5, reg_inc_beta_args)/vol_
    return(kernel_mat, mmd_mat)

def rbf_kernel(distance_matrix:np.ndarray, bandwidth:float, dim:int):
    dmat = np.square(distance_matrix)/bandwidth

    kernel_mat = np.power(1./(2.*np.pi*bandwidth), 0.5*dim)*np.exp(-0.5*dmat)
    mmd_mat = np.power(1./(2.*np.pi*bandwidth), 0.5*dim)*np.exp(-0.25*dmat)
    return(kernel_mat, mmd_mat)

'''
    Class for computing kernel density estimates (and corresponding kernel matrices)
'''
class KDE():
    def __init__(self, 
                 X: np.ndarray,
                 bandwidth: float,
                 method: str = 'rbf'): 
        n, self.p = X.shape
        self.method = method
        self.distance_matrix = squareform(pdist(X, metric='euclidean'))

        if self.method == 'hat': ## hat kernel (bandwidth = radius of ball)
            self.kernel_mat, self.mmd_mat = hat_kernel(self.distance_matrix, bandwidth, self.p)
        else: ## rbf kernel (bandwidth = variance of Gaussian)
            self.kernel_mat, self.mmd_mat = rbf_kernel(self.distance_matrix, bandwidth, self.p)
        
        self.rowsums = np.sum(self.kernel_mat, axis=1)
        self.normed_kernel_mat = self.kernel_mat/self.rowsums[:,np.newaxis]
        self.U, self.S, self.Vt = np.linalg.svd(self.normed_kernel_mat)
    
    def reset_bandwidth(self, bandwidth):
        if self.method == 'hat': ## hat kernel (bandwidth = radius of ball)
            self.kernel_mat, self.mmd_mat = hat_kernel(self.distance_matrix, bandwidth, self.p)
        else: ## rbf kernel (bandwidth = variance of Gaussian)
            self.kernel_mat, self.mmd_mat = rbf_kernel(self.distance_matrix, bandwidth, self.p)

        self.rowsums = np.sum(self.kernel_mat, axis=1)
        self.normed_kernel_mat = self.kernel_mat/self.rowsums[:,np.newaxis]
        self.U, self.S, self.Vt = np.linalg.svd(self.normed_kernel_mat)

    '''
        U, S, Vt: SVD of the row NORMALIZED kernel matrix: A_{ij} = K(x_i, x_j) / [sum_k K(x_i, x_k) ]
    '''
    def normalized_svd(self):
        return(self.U, self.S, self.Vt)

    '''
       A: row NORMALIZED kernel matrix: A_{ij} = K(x_i, x_j) / [sum_k K(x_i, x_k) ]
    '''
    def normalized_kernel_mat(self):
        return(self.normed_kernel_mat)

    '''
        rowsums: row sums of the kernel matrix: v_i = sum_k K(x_i, x_k)
    '''
    def row_sums(self):
        return(self.rowsums)


    def log_likelihood(self):
        return(np.log(self.rowsums))

    '''
        MMD: corresponding MMD matrix for the kernel -- MMD_{ij} =  \int_x K(x_i, x) K(x_j, x) dx
    '''
    def mmd_matrix(self):
        return(self.mmd_mat)

