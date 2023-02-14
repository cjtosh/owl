import abc
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, betainc
from sklearn.neighbors import NearestNeighbors


## Fast bandwidth selection
def knn_bandwidth(X, k):
    nbrs = NearestNeighbors().fit(X)
    ds, _ = nbrs.kneighbors(X, n_neighbors=k)
    return(np.median(ds[:,-1]))


'''
    Class for computing kernel density estimates (and corresponding kernel matrices)
'''
class KDE():
    __metaclass__ = abc.ABCMeta
    def __init__(self, 
                 X: np.ndarray,
                 bandwidth:float=None,
                 neighbors:int=None,
                 **kwargs): 

        if bandwidth is not None:
            self.bandwidth = bandwidth
        elif neighbors is not None:
            self.bandwidth = knn_bandwidth(X, k=neighbors)
        else:
            raise ValueError("One of bandwidth or neighbors must be set")

        n, self.dim = X.shape
        self.distance_matrix = squareform(pdist(X, metric='euclidean'))
        self.recalculate_kernel(self.bandwidth, **kwargs)
        # self.kernel_mat, self.mmd_mat = self.kernel_and_mmd_matrix(**kwargs)
        # self.rowsums = np.sum(self.kernel_mat, axis=1)
        # self.normed_kernel_mat = self.kernel_mat/self.rowsums[:,np.newaxis]
        # self.U, self.S, self.Vt = np.linalg.svd(self.normed_kernel_mat)

    @abc.abstractmethod
    def kernel_and_mmd_matrix(self, **kwargs) -> tuple(np.ndarray, np.ndarray):
        raise NotImplementedError

    
    def recalculate_kernel(self, bandwidth:float, **kwargs):
        self.bandwidth = bandwidth
        self.kernel_mat, self.mmd_mat = self.kernel_and_mmd_matrix(**kwargs)
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


class RBFKDE(KDE):
    def __init__(self, 
                X: np.ndarray,
                bandwidth: float):
    
        super().__init__(X=X, bandwidth=bandwidth)
    
    def kernel_and_mmd_matrix(self, **kwargs) -> tuple(np.ndarray, np.ndarray):
        dmat = np.square(self.distance_matrix)/self.bandwidth

        kernel_mat = np.power(1./(2.*np.pi*self.bandwidth), 0.5*self.dim)*np.exp(-0.5*dmat)
        mmd_mat = np.power(1./(2.*np.pi*self.bandwidth), 0.5*self.dim)*np.exp(-0.25*dmat)
        return(kernel_mat, mmd_mat)

    

class HatKDE(KDE):
    def __init__(self, 
                X: np.ndarray,
                bandwidth: float):
    
        super().__init__(X=X, bandwidth=bandwidth)
    
    def kernel_and_mmd_matrix(self, **kwargs) -> tuple(np.ndarray, np.ndarray):
        indicator = (self.distance_matrix <= self.bandwidth).astype(float)

        ## Volume of d-dimensional ball of radius bandwidth
        vol_ = np.power(np.pi, 0.5*self.dim)*np.power(self.bandwidth, self.dim)/gamma(0.5*self.dim + 1.) 
        kernel_mat = indicator/vol_

        reg_inc_beta_args = indicator*(1.0 - 0.25*np.square(self.distance_matrix/self.bandwidth))
        mmd_mat = betainc( 0.5*(self.dim + 1.), 0.5, reg_inc_beta_args)/vol_
        return(kernel_mat, mmd_mat)