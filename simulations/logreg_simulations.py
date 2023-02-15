import os
import pickle
import numpy as np
from scipy.special import expit
from scipy.io import loadmat
import argparse
from owl.regression import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV, RANSACRegressor
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from owl.ball import L1Ball


def logreg_corruption_comparison(X_train_:np.ndarray, y_train_:np.ndarray, X_test_:np.ndarray, y_test_:np.ndarray, epsilon, corr_type):
    results = []
    X_train = X_train_.copy()
    y_train = y_train_.copy()
    X_test = X_test_.copy()
    y_test = y_test_.copy()
    
    n_train, p = X_train.shape
    n_corrupt = int(epsilon*n_train)
    lr = LogisticRegression(X=X_train, y=y_train)
    lr.EM_step()
    train_acc = np.mean(lr.predict(X_train) == y_train)
    test_acc = np.mean(lr.predict(X_test) == y_test)
    results.append({"Method": "Uncorrupted MLE", 
                    "Corruption fraction": epsilon, 
                    "Test accuracy": test_acc,
                    "Train accuracy": train_acc,
                    "Corruption type": corr_type})

    if corr_type=='max':
        lls = lr.log_likelihood_vector() ## Get likelihood values
        inds_corrupt = np.argsort(-lls)[:n_corrupt] ## Corrupt largest indices
    else:
        inds_corrupt = np.random.choice(n_train, size=n_corrupt, replace=False)
    y_train[inds_corrupt] = (y_train[inds_corrupt] == 0).astype(int) ## Flip these labels

    ## MLE
    lr = LogisticRegression(X=X_train, y=y_train)
    lr.maximize_weighted_likelihood()
    train_acc = np.mean(lr.predict(X_train) == y_train)
    test_acc = np.mean(lr.predict(X_test) == y_test)
    results.append({"Method": "MLE", 
                    "Corruption fraction": epsilon, 
                    "Test accuracy": test_acc,
                    "Train accuracy": train_acc,
                    "Corruption type": corr_type})
    
    ## Robust logistic regression   
    rob_lr = LogisticRegression(X=X_train, y=y_train)
    l1_ball = L1Ball(n=n_train, r=2*epsilon)
    rob_lr.fit_owl(ball=l1_ball, n_iters=10)
    
    train_acc = np.mean( rob_lr.predict(X_train) == y_train)
    test_acc = np.mean( rob_lr.predict(X_test) == y_test)
    results.append({"Method": "OWL (TV)", 
                    "Corruption fraction": epsilon, 
                    "Test accuracy": test_acc,
                    "Train accuracy": train_acc,
                    "Corruption type": corr_type})


    ## Scale the data
    scaler = StandardScaler()
    scaler.fit(X=X_train)
    X_scale = scaler.transform(X=X_train)
    X_test_scale = scaler.transform(X=X_test)

    ## Logistic regression with cross validation
    clf = LogisticRegressionCV(cv=3)
    clf.fit(X=X_scale, y=y_train)
    train_acc = np.mean( clf.predict(X_scale) == y_train)
    test_acc = np.mean( clf.predict(X_test_scale) == y_test)
    results.append({"Method": "Regularized MLE (CV)", 
                    "Corruption fraction": epsilon, 
                    "Test accuracy": test_acc,
                    "Train accuracy": train_acc,
                    "Corruption type": corr_type})

    ## RANSAC regression
    clf = RANSACRegressor(estimator=LogReg(penalty='none'), min_samples=(1-2*epsilon), max_trials=50)
    clf.fit(X=X_scale, y=y_train)
    train_acc = np.mean( clf.predict(X_scale) == y_train)
    test_acc = np.mean( clf.predict(X_test_scale) == y_test)
    results.append({"Method": "RANSAC MLE", 
                    "Corruption fraction": epsilon, 
                    "Test accuracy": test_acc,
                    "Train accuracy": train_acc,
                    "Corruption type": corr_type})

    return(results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing arguments')
    parser.add_argument('--seed', type=int, default=100, help="The random seed.")
    parser.add_argument('--dataset', type=str, default='mnist', help="The dataset we're looking at.")
    parser.add_argument('--corr_type', type=str, default='max', help="The type of corruption.")

    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    corr_type = args.corr_type

    folder = os.path.join("results/logistic_regression", args.dataset)
    os.makedirs(folder, exist_ok=True)

    if args.dataset=='mnist':
        with open("data/mnist.pkl", 'rb') as io:
            mnist = pickle.load(io)

        x_ = mnist.data
        y_ = mnist.target
        mask = (y_=='1') | (y_=='8')
        X = x_[mask].values
        y = (y_[mask].values == '1').astype(int)

        test_frac = 0.2
        N = len(y)
        n_test = int(test_frac * N)
        test_inds = np.random.choice(N, size=n_test, replace=False)
        train_mask = ~np.isin(np.arange(N), test_inds)
        y_train = y[train_mask]
        y_test = y[~train_mask]

        ## PCA
        pca = PCA(n_components=10)
        pca.fit(X[train_mask])
        X_train = pca.transform(X[train_mask])
        X_test = pca.transform(X[~train_mask])
    elif args.dataset=='enron':
        enron = loadmat('data/enron_data.mat')
        X = np.vstack((np.asarray(enron['X_train'].todense()), np.asarray(enron['X_test'].todense())))
        y = np.concatenate((np.squeeze(enron['y_train']), np.squeeze(enron['y_test'])))
        y = (y > 0).astype(int)
        
        ## Make our own train-test split
        test_frac = 0.2
        N = len(y)
        n_test = int(test_frac * N)
        test_inds = np.random.choice(N, size=n_test, replace=False)
        train_mask = ~np.isin(np.arange(N), test_inds)
        y_train = y[train_mask]
        y_test = y[~train_mask]

        ## PCA
        pca = PCA(n_components=10)
        pca.fit(X[train_mask])
        X_train = pca.transform(X[train_mask])
        X_test = pca.transform(X[~train_mask])
    elif args.dataset=='random':
        stdv = 2.0
        n = 1000
        p = 10
        w = stdv*np.random.randn(p)

        X_train = np.random.randn(n, p)
        probs = expit(np.dot(X_train, w))
        z = np.random.rand(n)
        y_train = (z < probs).astype(int)

        X_test = np.random.randn(n, p)
        y_test = (np.dot(X_test, w) > 0).astype(int)

    full_results = []
    for epsilon in np.linspace(start=0.01, stop=0.25, num=10):
        results = logreg_corruption_comparison(X_train, y_train, X_test, y_test, epsilon, corr_type)
        full_results.extend(results)

    fname = os.path.join(folder, corr_type+"_"+str(seed)+".pkl")
    with open(fname, 'wb') as io:
        pickle.dump(full_results, io)

    