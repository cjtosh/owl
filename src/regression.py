import numpy as np
from cmodels import CModel
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.preprocessing import StandardScaler


class LinearRegression(CModel):
    def __init__(self, X:np.ndarray, y:np.ndarray, hard:bool=True, w:np.ndarray=None):
        self.y = y
        self.X = X
        n, self.p = X.shape
        super().__init__(n=n, w=w, hard=hard)
        self.clf = LinReg()

    def EM_step(self, n_steps: int = 1, hard: bool = True):
        if n_steps > 0:
            self.clf.fit(X=self.X, y=self.y, sample_weight=self.w)
            self.resid = self.y - self.predict(self.X)
            self.sigma = np.sqrt(np.sum(self.w*np.square(self.resid))/self.n)

    def log_likelihood_vector(self):
        return(stats.norm.logpdf(self.resid, 0, self.sigma))

    def predict(self, X:np.ndarray):
        return(self.clf.predict(X))

    def r2_score(self, X:np.ndarray, y:np.ndarray):
        return(self.clf.score(X=X, y=y))


class LogisticRegression(CModel):
    def __init__(self, X:np.ndarray, y:np.ndarray, hard:bool=True, w:np.ndarray=None):
        assert np.all(np.isin(y, [0,1])), "Values must be either 0 or 1"
        self.y = y
        n, self.p = X.shape
        super().__init__(n=n, w=w, hard=hard)
        self.scaler = StandardScaler()
        self.scaler.fit(X=X)
        self.X_scaled = self.scaler.transform(X=X)
        self.clf = LogReg(penalty='none', max_iter=300)

    def EM_step(self, n_steps: int = 1, hard: bool = True):
        if n_steps > 0:
            self.clf.fit(X=self.X_scaled, y=self.y, sample_weight=self.w)

    def log_likelihood_vector(self):
        prob_matrix = self.clf.predict_proba(self.X_scaled)
        probs = np.clip(prob_matrix[np.arange(self.n),self.y], a_min=10e-300, a_max=None) ## Stability
        ll = np.log(probs)
        return(ll)

    def predict(self, X:np.ndarray):
        X_s = self.scaler.transform(X=X)
        return( self.clf.predict(X_s))
