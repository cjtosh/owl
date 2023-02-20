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
        self.recalculate_kernel(self.bandwidth)


    '''
        Should set self.kernel_mat
    '''
    @abc.abstractmethod
    def calculate_kernel_matrix(self, **kwargs):
        raise NotImplementedError

    '''
        Should set self.mmd_mat
    '''
    @abc.abstractmethod
    def calculate_mmd_matrix(self, **kwargs):
        raise NotImplementedError

    
    def recalculate_kernel(self, bandwidth:float, **kwargs):
        self.bandwidth = bandwidth
        self.rowsums = None
        self.kernel_mat = None
        self.mmd_mat = None
        self.normed_kernel_mat = None
        self.U, self.S, self.Vt = None, None, None


    def calculate_normalized_kernel_matrix(self, **kwargs):
        if self.kernel_mat is None:
            self.calculate_kernel_matrix()
        
        rowsums = self.row_sums()
        
        self.normed_kernel_mat = self.kernel_mat/rowsums[:,np.newaxis]

    '''
        U, S, Vt: SVD of the row NORMALIZED kernel matrix: A_{ij} = K(x_i, x_j) / [sum_k K(x_i, x_k) ]
    '''
    def normalized_svd(self):
        if (self.U is None) or (self.S is None) or (self.Vt is None):
            if self.normed_kernel_mat is None:
                self.calculate_normalized_kernel_matrix()
            
            self.U, self.S, self.Vt = np.linalg.svd(self.normed_kernel_mat)
        return(self.U, self.S, self.Vt)

    '''
       A: row NORMALIZED kernel matrix: A_{ij} = K(x_i, x_j) / [sum_k K(x_i, x_k) ]
    '''
    def normalized_kernel_matrix(self):
        if self.normed_kernel_mat is None:
            self.calculate_normalized_kernel_matrix()
        return(self.normed_kernel_mat)

    '''
        rowsums: row sums of the kernel matrix: v_i = sum_k K(x_i, x_k)
    '''
    def row_sums(self):
        if self.rowsums is None:
            if self.kernel_mat is None:
                self.calculate_kernel_matrix()
            self.rowsums = np.sum(self.kernel_mat, axis=1)
        return(self.rowsums)

    def log_likelihood(self):
        rowsums = self.row_sums()
        return(np.log(rowsums))

    '''
        MMD: corresponding MMD matrix for the kernel -- MMD_{ij} =  \int_x K(x_i, x) K(x_j, x) dx
    '''
    def mmd_matrix(self):
        if self.mmd_mat is None:
            self.calculate_mmd_matrix()
        return(self.mmd_mat)


class RBFKDE(KDE):
    def __init__(self, 
                 X: np.ndarray,
                 bandwidth:float=None,
                 neighbors:int=None,
                 **kwargs): 
    
        super().__init__(X=X, bandwidth=bandwidth, neighbors=neighbors, **kwargs)
    
    def calculate_mmd_matrix(self, **kwargs):
        dmat = np.square(self.distance_matrix)/self.bandwidth
        self.mmd_mat = np.power(1./(2.*np.pi*self.bandwidth), 0.5*self.dim)*np.exp(-0.25*dmat)

    
    def calculate_kernel_matrix(self, **kwargs):
        dmat = np.square(self.distance_matrix)/self.bandwidth
        self.kernel_mat = np.power(1./(2.*np.pi*self.bandwidth), 0.5*self.dim)*np.exp(-0.5*dmat)

    
class HatKDE(KDE):
    def __init__(self, 
                 X: np.ndarray,
                 bandwidth:float=None,
                 neighbors:int=None,
                 **kwargs): 
    
        super().__init__(X=X, bandwidth=bandwidth, neighbors=neighbors, **kwargs)
    
    def calculate_kernel_matrix(self, **kwargs):
        indicator = (self.distance_matrix <= self.bandwidth).astype(float)

        ## Volume of d-dimensional ball of radius bandwidth
        vol_ = np.power(np.pi, 0.5*self.dim)*np.power(self.bandwidth, self.dim)/gamma(0.5*self.dim + 1.) 
        self.kernel_mat = indicator/vol_

    def calculate_mmd_matrix(self, **kwargs):
        vol_ = np.power(np.pi, 0.5*self.dim)*np.power(self.bandwidth, self.dim)/gamma(0.5*self.dim + 1.) 
        indicator = (self.distance_matrix <= self.bandwidth).astype(float)
        reg_inc_beta_args = indicator*(1.0 - 0.25*np.square(self.distance_matrix/self.bandwidth))
        self.mmd_mat = betainc( 0.5*(self.dim + 1.), 0.5, reg_inc_beta_args)/vol_