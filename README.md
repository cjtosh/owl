# OWL: Robustifying likelihoods by optimistically re-weighting data

This package implements OWL, a robust approach for fitting probabilistic models with likelihood functions.

## Installing OWL

If you have conda installed, then you can install by running the following from the base directory.

```
conda env create -f env.yaml
conda activate owl
pip install -e .
```

Otherwise, you should install the packages listed in `env.yaml`.

## Running OWL

To fit a probabilistic model using OWL, create a class that extends the `OWLModel` class. You must implement two functions: `maximize_weighted_likelihood` and `log_likelihood`. Below is a simple exponential distribution. 

```python
from owl.models import OWLModel

'''
    Simple univariate exponential distribution.
'''
class Exponential(OWLModel):
    def __init__(self, 
        X: np.ndarray, ## Input samples (1-dimensional)
        w:np.ndarray = None, ## Weights over the samples (set to None for uniform)
        **kwargs
        ):
        self.X = X.copy()
        n  = len(X)
        super().__init__(n=n, w=w, **kwargs)

        self.lam = 1.0 ## Parameter of the exponential distribution
    
    def maximize_weighted_likelihood(self, **kwargs):
        self.lam = np.sum(self.w)/np.dot(self.w, self.X)

    def log_likelihood(self):
        return( np.log(self.lam) - (self.lam*self.X) )
```

Once the class is created, then we need to choose the `Ball` class that we will fit it with. In all the experiments in the paper, the `L1Ball` class is used. 

```python
from owl.ball import L1Ball

## Generate data from an exponential distribution
n = 1000
lam = 5.0
x = np.random.exponential(scale=(1./lam), size=n)

## Randomly corrupt 5 percent of the data
epsilon = 0.05
corrupt_inds = np.random.choice(n, size=int(n*epsilon), replace=False)
for i in corrupt_inds:
    x[i] = 5.0 + np.random.standard_normal()
   
## Fit an owl estimate to the data
owl = Exponential(X=x)
l1ball = L1Ball(n=n, r=epsilon)
owl.fit_owl(ball=l1ball, n_iters=10, verbose=True)
```

More examples are in the `examples/Simple OWL models.ipynb` notebook.

## Citation

If you use this code, please cite the [preprint]( missing ):

```
Robustifying likelihoods by optimistically re-weighting data
M. Dewaskar, C.Tosh, J. Knoblauch, and D. Dunson
blank
```