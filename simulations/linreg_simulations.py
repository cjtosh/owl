import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from owl.models import fit_owl
from owl.ball import L1Ball
from sklearn.linear_model import RANSACRegressor, RidgeCV, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from owl.regression import LinearRegression

ADMMSTEPS = 10000

def linreg_corruption_comparison(X_train_:np.ndarray, y_train_:np.ndarray, X_test_:np.ndarray, y_test_:np.ndarray, epsilon, corr_type):
    results = []
    X_train = X_train_.copy()
    y_train = y_train_.copy()
    X_test = X_test_.copy()
    y_test = y_test_.copy()
    
    n_train, p = X_train.shape
    n_corrupt = int(epsilon*n_train)
    lr = LinearRegression(X=X_train, y=y_train)
    lr.maximize_weighted_likelihood()

    train_mse = np.mean( np.square(lr.predict(X_train) - y_train))
    test_mse = np.mean( np.square(lr.predict(X_test) - y_test))
    r2 = lr.r2_score(X_test, y_test)

    resids = y_train - lr.predict(X_train)
    v = np.max(np.abs(y_train))

    results.append({"Method": "Uncorrupted MLE", 
                    "Corruption fraction": epsilon, 
                    "Test MSE": test_mse,
                    "Train MSE": train_mse,
                    "Test R^2": r2,
                    "Corruption type": corr_type})

    if corr_type=='max':
        lls = lr.log_likelihood() ## Get likelihood values
        inds_corrupt = np.argsort(-lls)[:n_corrupt] ## Corrupt largest indices
    else:
        inds_corrupt = np.random.choice(n_train, size=n_corrupt, replace=False)
    
    resid_corrupt = resids[inds_corrupt]
    y_train[inds_corrupt] = 3*np.where(resid_corrupt>0, v, -v)


    ## MLE
    lr = LinearRegression(X=X_train, y=y_train)
    lr.maximize_weighted_likelihood()
    train_mse = np.mean( np.square(lr.predict(X_train) - y_train))
    test_mse = np.mean( np.square(lr.predict(X_test) - y_test))
    r2 = lr.r2_score(X_test, y_test)
    
    results.append({"Method": "MLE", 
                    "Corruption fraction": epsilon, 
                    "Test MSE": test_mse,
                    "Train MSE": train_mse,
                    "Test R^2": r2,
                    "Corruption type": corr_type})
    

    ## OWL with TV distance (search for radius)   
    rob_lr = LinearRegression(X=X_train, y=y_train)
    l1_ball = L1Ball(n=n_train, r=1.0)
    rob_lr = fit_owl(rob_lr, 
                     l1_ball, 
                     epsilons=np.linspace(0.01, 0.5, 20), 
                     admmsteps=ADMMSTEPS,
                     n_workers=4)
    
    train_mse = np.mean( np.square(rob_lr.predict(X_train) - y_train))
    test_mse = np.mean( np.square(rob_lr.predict(X_test) - y_test))
    r2 = rob_lr.r2_score(X_test, y_test)

    results.append({"Method": "OWL (TV)", 
                    "Corruption fraction": epsilon, 
                    "Test MSE": test_mse,
                    "Train MSE": train_mse,
                    "Test R^2": r2,
                    "Corruption type": corr_type})
    

    ## OWL with TV distance (known radius)   
    rob_lr = LinearRegression(X=X_train, y=y_train)
    l1_ball = L1Ball(n=n_train, r=2*epsilon)
    rob_lr.fit_owl(ball=l1_ball, admmsteps=ADMMSTEPS)
    
    train_mse = np.mean( np.square(rob_lr.predict(X_train) - y_train))
    test_mse = np.mean( np.square(rob_lr.predict(X_test) - y_test))
    r2 = rob_lr.r2_score(X_test, y_test)

    results.append({"Method": "OWL ($\epsilon$ known)", 
                    "Corruption fraction": epsilon, 
                    "Test MSE": test_mse,
                    "Train MSE": train_mse,
                    "Test R^2": r2,
                    "Corruption type": corr_type})


    ## Scale the data
    scaler = StandardScaler()
    scaler.fit(X=X_train)
    X_scale = scaler.transform(X=X_train)
    X_test_scale = scaler.transform(X=X_test)

    ## Ridge regression with cross validation
    clf = RidgeCV(cv=3)
    clf.fit(X=X_scale, y=y_train)
    
    train_mse = np.mean( np.square(clf.predict(X_scale) - y_train))
    test_mse = np.mean( np.square(clf.predict(X_test_scale) - y_test))
    r2 = clf.score(X_test_scale,y_test)
    results.append({"Method": "Ridge regression (CV)", 
                    "Corruption fraction": epsilon, 
                    "Test MSE": test_mse,
                    "Train MSE": train_mse,
                    "Test R^2": r2,
                    "Corruption type": corr_type})

    ## RANSAC regression
    clf = RANSACRegressor(min_samples=(1-2*epsilon), max_trials=50)
    clf.fit(X=X_scale, y=y_train)
    train_mse = np.mean( np.square(clf.predict(X_scale) - y_train))
    test_mse = np.mean( np.square(clf.predict(X_test_scale) - y_test))
    r2 = clf.score(X_test_scale,y_test)
    results.append({"Method": "RANSAC MLE", 
                    "Corruption fraction": epsilon, 
                    "Test MSE": test_mse,
                    "Train MSE": train_mse,
                    "Test R^2": r2,
                    "Corruption type": corr_type})

    ## Huber regression
    clf = HuberRegressor(epsilon=1.345)
    clf.fit(X=X_scale, y=y_train)

    train_mse = np.mean( np.square(clf.predict(X_scale) - y_train))
    test_mse = np.mean( np.square(clf.predict(X_test_scale) - y_test))
    r2 = clf.score(X_test_scale,y_test)
    results.append({"Method": "Huber Regression", 
                    "Corruption fraction": epsilon, 
                    "Test MSE": test_mse,
                    "Train MSE": train_mse,
                    "Test R^2": r2,
                    "Corruption type": corr_type})
    return(results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear regression simulations')
    parser.add_argument('--seed', type=int, default=100, help="The random seed.")
    parser.add_argument('--dataset', type=str, default='qsar', help="The dataset to run on (input 'qsar' or 'random').")
    parser.add_argument('--corr_type', type=str, default='max', help="The method of choosing corruptions (input 'max' or 'rand').")

    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    corr_type = args.corr_type

    folder = os.path.join("results/linear_regression", args.dataset)
    os.makedirs(folder, exist_ok=True)

    if args.dataset=='qsar':
        df = pd.read_csv("data/qsar.csv")
        y = df['pXC50'].values
        df.drop(columns=['target_id', 'molecule_id', 'pXC50', 'dataset_id'], inplace=True)
        X = df.values

        ## Make train-test split
        test_frac = 0.2
        N = len(y)
        n_test = int(test_frac * N)
        test_inds = np.random.choice(N, size=n_test, replace=False)
        train_mask = ~np.isin(np.arange(N), test_inds)
        y_train = y[train_mask]
        y_test = y[~train_mask]
        
        pca = PCA(n_components=50)
        pca.fit(X[train_mask])
        X_train = pca.transform(X[train_mask])
        X_test = pca.transform(X[~train_mask])
    elif args.dataset=='random':
        stdv_0 = 2.0
        stdv = 0.25
        n = 5000
        p = 10
        w = stdv_0*np.random.randn(p)

        X_train = np.random.randn(n, p)
        y_train = np.dot(X_train, w) + stdv*np.random.randn(n)

        X_test = np.random.randn(n, p)
        y_test = np.dot(X_test, w)

    full_results = []
    epsilons =  np.linspace(start=0.01, stop=0.25, num=10)
    for epsilon in tqdm(epsilons):
        results = linreg_corruption_comparison(X_train, y_train, X_test, y_test, epsilon, corr_type)
        full_results.extend(results)

    fname = os.path.join(folder, corr_type+"_"+str(seed)+".pkl")
    with open(fname, 'wb') as io:
        pickle.dump(full_results, io)

    