import abc
import numpy as np
from numpy.linalg import norm, multi_dot
from proximal import ProxL1Ball, ProxL2Ball, ProxMMDBall, proj_l1_ball, proj_l2_ball, proj_ellipsoid
from typing import Optional


'''
    Class that implements projection onto a probability ball with respect to some distance. 
'''
class ProbabilityBall():
    __metaclass__ = abc.ABCMeta
    def __init__(self, 
                 n:int, ## Dimension of the probability vector
                 r:float, ## Radius of the ball
                 center:Optional[np.ndarray]=None, ## OPTIONAL: Center of the probability vector. If None, then use (1/n, ..., 1/n)
                 **kwargs
                 ):
        self.n = n
        self.r = r
        if center is None:
            self.center = np.ones(n)/float(n)
        else:
            self.center = center
        
            
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


class MMDBall(ProbabilityBall):
    def __init__(self, 
                n:int, ## Dimension of the probability vector
                r:float, ## Radius of the ball
                kernel_matrix:np.ndarray, ## MMD kernel matrix
                center:Optional[np.ndarray]=None, ## OPTIONAL: Center of the probability vector. If None, then use (1/n, ..., 1/n)
                **kwargs
                ):
            super().__init__(n=n, r=r, center=center, **kwargs)

            a, b = kernel_matrix.shape
            assert ((a==n) and (b==n)), "Error -- kernel matrix has shape " + str(a) + " x " + str(b) + ", expected shape "  + str(n) + " x " + str(n)
            eigL, self.Q = np.linalg.eigh(kernel_matrix)
            self.eigL = np.clip(eigL, a_min=0.0, a_max=None) ## Clip eigenvalues at 0



    def projection(self, x: np.ndarray):
        return(proj_ellipsoid(c=x, mu=self.center, Q=self.Q, L=self.eigL, T=100, r=self.r))


    def get_prox_operator(self, tilt:bool=False):
        return(ProxMMDBall(center=self.center, eigL=self.eigL, Q=self.Q, r=self.r, tilt=tilt))

    def projection_value(self, x:np.ndarray):
        diff =  x - self.center
        return( multi_dot([ diff.T, self.Q, np.diag(self.eigL), self.Q.T, diff]) )

    
class L1Ball(ProbabilityBall):
    def __init__(self, 
                n:int, ## Dimension of the probability vector
                r:float, ## Radius of the ball
                center:Optional[np.ndarray]=None, ## OPTIONAL: Center of the probability vector. If None, then use (1/n, ..., 1/n)
                **kwargs
                ):
            super().__init__(n=n, r=r, center=center, **kwargs)

    def projection(self, x: np.ndarray):
        return(proj_l1_ball(x=x, c=self.center, r=self.r))


    def get_prox_operator(self, tilt:bool=False):
        return(ProxL1Ball(c=self.center, r=self.r, tilt=tilt))

    def projection_value(self, x:np.ndarray):
        diff =  x - self.center
        return( norm(diff, ord=1) )

class L2Ball(ProbabilityBall):
    def __init__(self, 
                n:int, ## Dimension of the probability vector
                r:float, ## Radius of the ball
                center:Optional[np.ndarray]=None, ## OPTIONAL: Center of the probability vector. If None, then use (1/n, ..., 1/n)
                **kwargs
                ):
            super().__init__(n=n, r=r, center=center, **kwargs)

    def projection(self, x: np.ndarray):
        return(proj_l2_ball(x=x, c=self.center, r=self.r))

    def get_prox_operator(self, tilt:bool=False):
        return(ProxL2Ball(c=self.center, r=self.r, tilt=tilt))

    def projection_value(self, x:np.ndarray):
        diff =  x - self.center
        return( norm(diff) )

