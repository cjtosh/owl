import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import os
from kneed import KneeLocator
from sklearn.cluster import KMeans
from owl.mixture_models import GeneralGMM
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

def load_results(folder):
    fnames = [x for x in os.listdir(folder) if x.endswith('.pkl')]
    rand_results = []
    max_results = []
    for fname in fnames:
        with open(os.path.join(folder, fname), 'rb') as io:
            res = pickle.load(io)

        if fname.startswith('max'):
            max_results.extend(res)
        else:
            rand_results.extend(res)
    max_df = pd.DataFrame(max_results)
    rand_df = pd.DataFrame(rand_results)
    max_df.replace("CMLE (TV)", "OWL", inplace=True)
    rand_df.replace("CMLE (TV)", "OWL", inplace=True)
    max_df.replace("OWL (TV)", "OWL", inplace=True)
    rand_df.replace("OWL (TV)", "OWL", inplace=True)
    max_df.replace("OWL (Kernelized - TV)", "OWL (Kernelized)", inplace=True)
    rand_df.replace("OWL (Kernelized - TV)", "OWL (Kernelized)", inplace=True)
    return(max_df, rand_df)


os.makedirs('figures', exist_ok=True)

colors = sns.color_palette("husl", 5)
logistic_orders = ['OWL', 'RANSAC MLE', 'Regularized MLE (CV)', 'MLE']
linear_orders = ['MLE', 'Ridge Regression (CV)', 'RANSAC MLE', 'Huber Regression', 'OWL']
sublinear_orders = ['RANSAC MLE', 'Huber Regression', 'OWL']


palette={'OWL':colors[0], 
         'MLE':colors[1], 
         'Regularized MLE (CV)':colors[4], 
         'RANSAC MLE':colors[3], 
         'Ridge Regression (CV)':colors[4], 
         'Huber Regression':colors[2],
         'OWL (Kernelized)':colors[2]}

size_2x3 = (16,8)
size_2x2 = (13,8)
size_1x3 = (13,4)
size_1x2 = (12,4)
fontsize_1x3 = 11
fontsize_2x3 = 15
fontsize_2x2 = 15


'''
    Logistic regression plots (Max-corruptions)
'''
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=size_1x3)

max_df, _ = load_results('results/logistic_regression/random/')
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Test accuracy'].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]


axs[0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=max_df, palette=palette, hue_order=logistic_orders, ax=axs[0])
axs[0].set_title('Simulated Data', size=fontsize_1x3)
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles=handles, labels=labels)



max_df, _ = load_results('results/logistic_regression/mnist/')
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Test accuracy'].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]

axs[1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=max_df, palette=palette, hue_order=logistic_orders, ax=axs[1])
axs[1].set_title('MNIST (1 v.s. 8)', size=fontsize_1x3)
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles=handles, labels=labels)


max_df, _ = load_results('results/logistic_regression/enron/')
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Test accuracy'].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]

axs[2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=max_df, palette=palette, hue_order=logistic_orders, ax=axs[2])
axs[2].set_title('Enron Spam', size=fontsize_1x3)
handles, labels = axs[2].get_legend_handles_labels()
axs[2].legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig('figures/log_max_corruption.pdf', bbox_inches='tight')


'''
    Logistic regression plots (Rand-corruptions)
'''
plt.clf()
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=size_1x3)

_, rand_df = load_results('results/logistic_regression/random/')
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Test accuracy'].mean()
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]


axs[0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=rand_df, palette=palette, hue_order=logistic_orders, ax=axs[0])
axs[0].set_title('Simulated Data', size=fontsize_1x3)
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles=handles, labels=labels)



_, rand_df = load_results('results/logistic_regression/mnist/')
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Test accuracy'].mean()
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]

axs[1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=rand_df, palette=palette, hue_order=logistic_orders, ax=axs[1])
axs[1].set_title('MNIST (1 v.s. 8)', size=fontsize_1x3)
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles=handles, labels=labels)


_, rand_df = load_results('results/logistic_regression/enron/')
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Test accuracy'].mean()
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]

axs[2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=rand_df, palette=palette, hue_order=logistic_orders, ax=axs[2])
axs[2].set_title('Enron Spam', size=fontsize_1x3)
handles, labels = axs[2].get_legend_handles_labels()
axs[2].legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig('figures/log_rand_corruption.pdf', bbox_inches='tight')


'''
    Logistic regression plots (Combined)
'''
plt.clf()
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=size_2x3)

max_df, rand_df = load_results('results/logistic_regression/random/')
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Test accuracy'].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]


axs[0,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=max_df, palette=palette, hue_order=logistic_orders, ax=axs[0,0])
axs[0,0].set_title('Simulated Data', size=fontsize_2x3)
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles=handles, labels=labels)


axs[1,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=rand_df, palette=palette, hue_order=logistic_orders, ax=axs[1,0])
handles, labels = axs[0,1].get_legend_handles_labels()
axs[1,0].legend(handles=handles, labels=labels)



max_df, rand_df = load_results('results/logistic_regression/mnist/')
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Test accuracy'].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]

axs[0,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=max_df, palette=palette, hue_order=logistic_orders, ax=axs[0,1])
axs[0,1].set_title('MNIST (1 v.s. 8)', size=fontsize_2x3)
handles, labels = axs[1,0].get_legend_handles_labels()
axs[0,1].legend(handles=handles, labels=labels)

axs[1,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=rand_df, palette=palette, hue_order=logistic_orders, ax=axs[1,1])
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].legend(handles=handles, labels=labels)



max_df, rand_df = load_results('results/logistic_regression/enron/')
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Test accuracy'].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]

axs[0,2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=max_df, palette=palette, hue_order=logistic_orders, ax=axs[0,2])
axs[0,2].set_title('Enron Spam', size=fontsize_2x3)
handles, labels = axs[0,2].get_legend_handles_labels()
axs[0,2].legend(handles=handles, labels=labels)

axs[1,2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Test accuracy", hue="Method", data=rand_df, palette=palette, hue_order=logistic_orders, ax=axs[1,2])
handles, labels = axs[1,2].get_legend_handles_labels()
axs[1,2].legend(handles=handles, labels=labels)


for ax, row in zip(axs[:,0], ['Max-likelihood corruption', 'Random corruption']):
    ax.annotate(row, (0, 0.5), xytext=(-50, 0), ha='right', va='center',
                size=15, rotation=90, xycoords='axes fraction',
                textcoords='offset points')
    

plt.tight_layout()
plt.savefig('figures/log_corruption.pdf', bbox_inches='tight')





##################
##################
##################

y = "Test MSE"
# y = "Test R^2" 


'''
    Linear regression plots (Max-corruptions)
'''

plt.clf()
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=size_2x2)

max_df, _ = load_results('results/linear_regression/random/')

epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"][y].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]


axs[0,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=max_df, palette=palette, hue_order=linear_orders, ax=axs[0,0])
axs[0,0].set_title('Simulated Data', size=fontsize_2x2)
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles=handles, labels=labels)

max_df = max_df[max_df['Method'].isin(['OWL', 'RANSAC MLE', 'Huber Regression'])]

axs[1,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=max_df, palette=palette, hue_order=sublinear_orders, ax=axs[1,0])
handles, labels = axs[1,0].get_legend_handles_labels()
axs[1,0].legend(handles=handles, labels=labels)


max_df, _ = load_results('results/linear_regression/qsar/')
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"][y].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]


axs[0,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=max_df, palette=palette, hue_order=linear_orders, ax=axs[0,1])
axs[0,1].set_title('QSAR', size=fontsize_2x2)
handles, labels = axs[0,1].get_legend_handles_labels()
axs[0,1].legend(handles=handles, labels=labels)


# max_df = max_df[max_df['Method'].isin(['OWL', 'RANSAC MLE', 'Huber Regression'])]
max_df = max_df[max_df['Method'].isin(['OWL', 'Huber Regression'])]


axs[1,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=max_df, palette=palette, hue_order=['Huber Regression', 'OWL'], ax=axs[1,1])
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].legend(handles=handles, labels=labels)


plt.tight_layout()
plt.savefig('figures/lin_max_corruption.pdf', bbox_inches='tight')



'''
    Linear regression plots (Rand-corruptions)
'''


plt.clf()
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=size_2x2)

_, rand_df = load_results('results/linear_regression/random/')

epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"][y].mean()
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]


axs[0,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=rand_df, palette=palette, hue_order=linear_orders, ax=axs[0,0])
axs[0,0].set_title('Simulated Data', size=fontsize_2x2)
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles=handles, labels=labels)

rand_df = rand_df[rand_df['Method'].isin(['OWL', 'RANSAC MLE', 'Huber Regression'])]

axs[1,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=rand_df, palette=palette, hue_order=sublinear_orders, ax=axs[1,0])
handles, labels = axs[1,0].get_legend_handles_labels()
axs[1,0].legend(handles=handles, labels=labels)


_, rand_df = load_results('results/linear_regression/qsar/')
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"][y].mean()
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]


axs[0,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=rand_df, palette=palette, hue_order=linear_orders, ax=axs[0,1])
axs[0,1].set_title('QSAR', size=fontsize_2x2)
handles, labels = axs[0,1].get_legend_handles_labels()
axs[0,1].legend(handles=handles, labels=labels)


# rand_df = rand_df[rand_df['Method'].isin(['OWL', 'RANSAC MLE', 'Huber Regression'])]
rand_df = rand_df[rand_df['Method'].isin(['OWL',  'Huber Regression'])]


axs[1,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=rand_df, palette=palette, hue_order=['Huber Regression', 'OWL'], ax=axs[1,1])
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].legend(handles=handles, labels=labels)


plt.tight_layout()
plt.savefig('figures/lin_rand_corruption.pdf', bbox_inches='tight')



'''
    Linear regression plots (Combined)
'''
plt.clf()
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20,8))

max_df, rand_df = load_results('results/linear_regression/random/')


epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"][y].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]


axs[0,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=max_df, palette=palette, hue_order=linear_orders, ax=axs[0,0])
axs[0,0].set_title('Simulated Data', size=15)
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles=handles, labels=labels)


axs[1,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=rand_df, palette=palette, hue_order=linear_orders, ax=axs[1,0])
handles, labels = axs[1,0].get_legend_handles_labels()
axs[1,0].legend(handles=handles, labels=labels)


max_df = max_df[max_df['Method'].isin(['OWL', 'RANSAC MLE', 'Huber Regression'])]
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'RANSAC MLE', 'Huber Regression'])]

axs[0,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=max_df, palette=palette, hue_order=sublinear_orders, ax=axs[0,1])
axs[0,1].set_title('Simulated Data', size=15)
handles, labels = axs[0,1].get_legend_handles_labels()
axs[0,1].legend(handles=handles, labels=labels)


axs[1,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=rand_df, palette=palette, hue_order=sublinear_orders, ax=axs[1,1])
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].legend(handles=handles, labels=labels)




max_df, rand_df = load_results('results/linear_regression/qsar/')
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"][y].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]


axs[0,2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=max_df, palette=palette, hue_order=linear_orders, ax=axs[0,2])
axs[0,2].set_title('QSAR', size=15)
handles, labels = axs[0,2].get_legend_handles_labels()
axs[0,2].legend(handles=handles, labels=labels)


axs[1,2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=rand_df, palette=palette, hue_order=linear_orders, ax=axs[1,2])
handles, labels = axs[1,2].get_legend_handles_labels()
axs[1,2].legend(handles=handles, labels=labels)


max_df = max_df[max_df['Method'].isin(['OWL', 'RANSAC MLE', 'Huber Regression'])]
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'RANSAC MLE', 'Huber Regression'])]

axs[0,3].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=max_df, palette=palette, hue_order=sublinear_orders, ax=axs[0,3])
axs[0,3].set_title('QSAR', size=15)
handles, labels = axs[0,3].get_legend_handles_labels()
axs[0,3].legend(handles=handles, labels=labels)


axs[1,3].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y=y, hue="Method", data=rand_df, palette=palette, hue_order=sublinear_orders, ax=axs[1,3])
handles, labels = axs[0,3].get_legend_handles_labels()
axs[1,3].legend(handles=handles, labels=labels)


for ax, row in zip(axs[:,0], ['Max-likelihood corruption', 'Random corruption']):
    ax.annotate(row, (0, 0.5), xytext=(-50, 0), ha='right', va='center',
                size=15, rotation=90, xycoords='axes fraction',
                textcoords='offset points')
    

plt.tight_layout()
plt.savefig('figures/lin_corruption.pdf', bbox_inches='tight')



'''
    GMM + BMM (max-corruptions)
'''

max_df, _ = load_results('results/gmm_simulation/dim_10')
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"].reset_index(drop=True)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=size_1x2)

axs[0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=max_df, palette=palette, ax=axs[0])
axs[0].set_ylabel("Mean Parameter MSE")
axs[0].set_title('Gaussian mixture model')
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles=handles, labels=labels)

max_df, rand_df = load_results('results/bmm_simulation/dim_100/')
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Parameter L1 distance'].mean()
max_df = max_df[max_df['Method']!="Uncorrupted MLE"]

axs[1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Parameter L1 distance", hue="Method", data=max_df, palette=palette, ax=axs[1])
axs[1].set_title('Bernoulli mixture model')
axs[1].set_ylabel("Mean Parameter MAE")
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig('figures/clustering_max_corruption.pdf', bbox_inches='tight')


'''
    GMM + BMM (rand-corruptions)
'''

_, rand_df = load_results('results/gmm_simulation/dim_10')
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"].reset_index(drop=True)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=size_1x2)

axs[0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=rand_df, palette=palette, ax=axs[0])
axs[0].set_ylabel("Mean Parameter MSE")
axs[0].set_title('Gaussian mixture model')
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles=handles, labels=labels)

_, rand_df = load_results('results/bmm_simulation/dim_100/')
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Parameter L1 distance'].mean()
rand_df = rand_df[rand_df['Method']!="Uncorrupted MLE"]

axs[1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Parameter L1 distance", hue="Method", data=rand_df, palette=palette, ax=axs[1])
axs[1].set_title('Bernoulli mixture model')
axs[1].set_ylabel("Mean Parameter MAE")
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig('figures/clustering_rand_corruption.pdf', bbox_inches='tight')




'''
    Gaussian plots (Max-corruptions)
'''
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=size_1x3)

max_df, _ = load_results('results/gaussian/dim_2/')
max_df = max_df[max_df['Corruption fraction']<=0.255]
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
max_df = max_df[max_df['Method'].isin(['OWL', 'OWL (Kernelized, adaptive)'])]
max_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

axs[0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=max_df, palette=palette, ax=axs[0])
axs[0].set_title('Dimension = 2', size=fontsize_1x3)
handles, labels = axs[0].get_legend_handles_labels()
axs[0].set_ylabel("Mean Parameter MSE")
axs[0].legend(handles=handles, labels=labels)


max_df, _ = load_results('results/gaussian/dim_25/')
max_df = max_df[max_df['Corruption fraction']<=0.255]
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
max_df = max_df[max_df['Method'].isin(['OWL', 'OWL (Kernelized, adaptive)'])]
max_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

axs[1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=max_df, palette=palette, ax=axs[1])
axs[1].set_title('Dimension = 25', size=fontsize_1x3)
handles, labels = axs[1].get_legend_handles_labels()
axs[1].set_ylabel("Mean Parameter MSE")
axs[1].legend(handles=handles, labels=labels)


max_df, _ = load_results('results/gaussian/dim_50/')
max_df = max_df[max_df['Corruption fraction']<=0.255]
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
max_df = max_df[max_df['Method'].isin(['OWL', 'OWL (Kernelized, adaptive)'])]
max_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

axs[2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=max_df, palette=palette, ax=axs[2])
axs[2].set_title('Dimension = 50', size=fontsize_1x3)
handles, labels = axs[2].get_legend_handles_labels()
axs[2].set_ylabel("Mean Parameter MSE")
axs[2].legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig('figures/gaussian_max_corruption.pdf', bbox_inches='tight')



'''
    Gaussian plots (Max-corruptions) with mle
'''
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=size_2x3)

max_df, _ = load_results('results/gaussian/dim_2/')
max_df = max_df[max_df['Corruption fraction']<=0.255]
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
max_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

## First plot with MLE
max_df = max_df[max_df['Method'].isin(['OWL', 'MLE', 'OWL (Kernelized)'])]
axs[0,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=max_df, palette=palette, ax=axs[0,0])
axs[0,0].set_title('Dimension = 2', size=fontsize_2x3)
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,0].set_ylabel("Mean Parameter MSE")
axs[0,0].legend(handles=handles, labels=labels)

## Now without MLE
max_df = max_df[max_df['Method'].isin(['OWL', 'OWL (Kernelized)'])]
axs[1,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=max_df, palette=palette, ax=axs[1,0])
handles, labels = axs[1,0].get_legend_handles_labels()
axs[1,0].set_ylabel("Mean Parameter MSE")
axs[1,0].legend(handles=handles, labels=labels)


max_df, _ = load_results('results/gaussian/dim_25/')
max_df = max_df[max_df['Corruption fraction']<=0.255]
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
max_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

## First plot with MLE
max_df = max_df[max_df['Method'].isin(['OWL', 'MLE', 'OWL (Kernelized)'])]
axs[0,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=max_df, palette=palette, ax=axs[0,1])
axs[0,1].set_title('Dimension = 25', size=fontsize_2x3)
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,1].set_ylabel("Mean Parameter MSE")
axs[0,1].legend(handles=handles, labels=labels)

## Now without MLE
max_df = max_df[max_df['Method'].isin(['OWL', 'OWL (Kernelized)'])]
axs[1,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=max_df, palette=palette, ax=axs[1,1])
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].set_ylabel("Mean Parameter MSE")
axs[1,1].legend(handles=handles, labels=labels)



max_df, _ = load_results('results/gaussian/dim_50/')
max_df = max_df[max_df['Corruption fraction']<=0.255]
epsilons = np.unique(max_df['Corruption fraction'])
uncmle = max_df[max_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
max_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

## First plot with MLE
max_df = max_df[max_df['Method'].isin(['OWL', 'MLE', 'OWL (Kernelized)'])]
axs[0,2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=max_df, palette=palette, ax=axs[0,2])
axs[0,2].set_title('Dimension = 50', size=fontsize_2x3)
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,2].set_ylabel("Mean Parameter MSE")
axs[0,2].legend(handles=handles, labels=labels)

## Now without MLE
max_df = max_df[max_df['Method'].isin(['OWL', 'OWL (Kernelized)'])]
axs[1,2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=max_df, palette=palette, ax=axs[1,2])
handles, labels = axs[1,2].get_legend_handles_labels()
axs[1,2].set_ylabel("Mean Parameter MSE")
axs[1,2].legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig('figures/gaussian_wmle_max_corruption.pdf', bbox_inches='tight')



'''
    Gaussian plots (Rand-corruptions)
'''
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=size_1x3)

_, rand_df = load_results('results/gaussian/dim_2/')
rand_df = rand_df[rand_df['Corruption fraction']<=0.255]
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'OWL (Kernelized, adaptive)'])]
rand_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

axs[0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=rand_df, palette=palette, ax=axs[0])
axs[0].set_title('Dimension = 2', size=fontsize_1x3)
handles, labels = axs[0].get_legend_handles_labels()
axs[0].set_ylabel("Mean Parameter MSE")
axs[0].legend(handles=handles, labels=labels)


_, rand_df = load_results('results/gaussian/dim_25/')
rand_df = rand_df[rand_df['Corruption fraction']<=0.255]
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'OWL (Kernelized, adaptive)'])]
rand_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

axs[1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=rand_df, palette=palette, ax=axs[1])
axs[1].set_title('Dimension = 25', size=fontsize_1x3)
handles, labels = axs[1].get_legend_handles_labels()
axs[1].set_ylabel("Mean Parameter MSE")
axs[1].legend(handles=handles, labels=labels)


_, rand_df = load_results('results/gaussian/dim_50/')
rand_df = rand_df[rand_df['Corruption fraction']<=0.255]
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'OWL (Kernelized, adaptive)'])]
rand_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

axs[2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=rand_df, palette=palette, ax=axs[2])
axs[2].set_title('Dimension = 50', size=fontsize_1x3)
handles, labels = axs[2].get_legend_handles_labels()
axs[2].set_ylabel("Mean Parameter MSE")
axs[2].legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig('figures/gaussian_rand_corruption.pdf', bbox_inches='tight')



'''
    Gaussian plots (Max-corruptions) with mle
'''
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=size_2x3)

_, rand_df = load_results('results/gaussian/dim_2/')
rand_df = rand_df[rand_df['Corruption fraction']<=0.255]
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
rand_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

## First plot with MLE
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'MLE', 'OWL (Kernelized)'])]
axs[0,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=rand_df, palette=palette, ax=axs[0,0])
axs[0,0].set_title('Dimension = 2', size=fontsize_2x3)
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,0].set_ylabel("Mean Parameter MSE")
axs[0,0].legend(handles=handles, labels=labels)

## Now without MLE
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'OWL (Kernelized)'])]
axs[1,0].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=rand_df, palette=palette, ax=axs[1,0])
handles, labels = axs[1,0].get_legend_handles_labels()
axs[1,0].set_ylabel("Mean Parameter MSE")
axs[1,0].legend(handles=handles, labels=labels)


_, rand_df = load_results('results/gaussian/dim_25/')
rand_df = rand_df[rand_df['Corruption fraction']<=0.255]
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
rand_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

## First plot with MLE
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'MLE', 'OWL (Kernelized)'])]
axs[0,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=rand_df, palette=palette, ax=axs[0,1])
axs[0,1].set_title('Dimension = 25', size=fontsize_2x3)
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,1].set_ylabel("Mean Parameter MSE")
axs[0,1].legend(handles=handles, labels=labels)

## Now without MLE
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'OWL (Kernelized)'])]
axs[1,1].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=rand_df, palette=palette, ax=axs[1,1])
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].set_ylabel("Mean Parameter MSE")
axs[1,1].legend(handles=handles, labels=labels)



_, rand_df = load_results('results/gaussian/dim_50/')
rand_df = rand_df[rand_df['Corruption fraction']<=0.255]
epsilons = np.unique(rand_df['Corruption fraction'])
uncmle = rand_df[rand_df['Method']=="Uncorrupted MLE"]['Mean MSE'].mean()
rand_df.replace('OWL (Kernelized, adaptive)', 'OWL (Kernelized)', inplace=True)

## First plot with MLE
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'MLE', 'OWL (Kernelized)'])]
axs[0,2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=rand_df, palette=palette, ax=axs[0,2])
axs[0,2].set_title('Dimension = 50', size=fontsize_2x3)
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,2].set_ylabel("Mean Parameter MSE")
axs[0,2].legend(handles=handles, labels=labels)

## Now without MLE
rand_df = rand_df[rand_df['Method'].isin(['OWL', 'OWL (Kernelized)'])]
axs[1,2].plot(epsilons, np.repeat(uncmle, len(epsilons)), color='black', linestyle='dashed', linewidth=2)
sns.lineplot(x="Corruption fraction", y="Mean MSE", hue="Method", data=rand_df, palette=palette, ax=axs[1,2])
handles, labels = axs[1,2].get_legend_handles_labels()
axs[1,2].set_ylabel("Mean Parameter MSE")
axs[1,2].legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig('figures/gaussian_wmle_rand_corruption.pdf', bbox_inches='tight')



'''
    RNA-Seq elbow plots
'''

folder = "results/rna_seq/"
fnames = [f for f in os.listdir(folder) if f.endswith(".pkl")]
results = []
for f in fnames:
    eps = float(f.split('_')[0])
    with open(os.path.join(folder, f), 'rb') as io:
        l = pickle.load(io)
    results.extend(l)
df = pd.DataFrame(results)

df["epsilon"] = pd.Categorical(df["epsilon"])

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

sns.lineplot(x='K', y='Weighted Log-likelihood', hue="epsilon",  data=df, ax=axs[0])
axs[0].set_xlabel('Number of clusters')

mean_df = df.groupby(['epsilon', 'K']).mean().reset_index()
epsilons = mean_df['epsilon'].unique()
Ks = np.sort(mean_df['K'].unique())
K_lookup = {K:i for i,K in enumerate(Ks)}
lookup = {eps:np.empty(len(Ks)) for eps in epsilons}

for index, row in mean_df.iterrows():
    lookup[ row['epsilon'] ][ K_lookup[row['K']]] = row['Weighted Log-likelihood']
    
res_kndl = []
knee_lookup = {}
for eps in lookup.keys():
    kneedle = KneeLocator(Ks, lookup[eps], S=1.0, curve="concave", direction="increasing")
    knee_lookup[eps] = (kneedle.knee, np.max(kneedle.y_difference))
    for K, y in zip(Ks, kneedle.y_difference):
        res_kndl.append({"K":K, "epsilon":eps, "Normalized differences":y})

        


df_kndl = pd.DataFrame(res_kndl)
curr_epsilons = [0.05, 0.2, 0.4, 0.6, 0.8]
df_curr = df_kndl[df_kndl["epsilon"].isin(curr_epsilons)]
# df_curr = df_kndl
df_curr["epsilon"] = pd.Categorical(df_curr["epsilon"])
sns.lineplot(x="K", y="Normalized differences", hue="epsilon", data=df_curr, ax=axs[1])
axs[1].set_xlabel('Number of clusters')


plt.tight_layout()
plt.savefig('figures/rna_elbow.pdf', bbox_inches='tight')


'''
    RNA-Seq comparison plots  
'''
X_proc = pd.read_csv("data/cell_data.csv", index_col=0)
groups = np.array([g.split('__')[1] for g in X_proc.index.values])
groups_no_b = np.array([x.split('_')[0] for x in groups])

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_proc)
ndata, _ = X_pca.shape

kmeans_aris = []
seeds = list(range(100, 130))
for seed in tqdm(seeds):
    kmeans = KMeans(n_clusters=7, random_state=seed)
    kmeans.fit(X_pca)
    kmeans_aris.append(adjusted_rand_score(groups_no_b, kmeans.labels_))

mle_aris = []
for seed in tqdm(seeds):
    np.random.seed(seed)
    gmm = GeneralGMM(X=X_pca, K=7)
    gmm.fit_mle()
    mle_aris.append(adjusted_rand_score(groups_no_b, gmm.z))



df_7 = (df[df["K"]==7].reset_index(drop=True)).sort_values("epsilon")
results = []
for v in kmeans_aris:
    results.append({"Method":"K-Means", "Adjusted Rand Index": v})
    
for v in mle_aris:
    results.append({"Method":"MLE", "Adjusted Rand Index": v})
    
for index, row in df_7.iterrows():
    meth = "OWL (" + chr(949) + "=" + str(row["epsilon"]) + ")"
    results.append({"Method":meth, "Adjusted Rand Index": row["Adjusted Rand Index"]})

df_comp = pd.DataFrame(results)

results =[]
for index, row in df_7.iterrows():
    meth = "OWL (" + chr(949) + "=" + str(row["epsilon"]) + ")"
    results.append({"Method":meth, "Adjusted Rand Index (inliers)": row["Adjusted Rand Index (subset)"], "Fraction of inliers": (row["Number of inliers"]/ndata ) })

df_owl = pd.DataFrame(results)

methods = df_comp["Method"].unique()
colors = sns.color_palette("husl", len(methods))
palette = dict(zip(methods, colors))

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

sns.barplot(x="Method", y="Adjusted Rand Index", data=df_comp,palette=palette, ax=axs[0])
axs[0].tick_params(axis='x', labelrotation = 75)
axs[0].xaxis.label.set_visible(False)
axs[0].set_ylim([0.4, 1])


sns.barplot(x="Method", y="Adjusted Rand Index (inliers)", data=df_owl,palette=palette, ax=axs[1])
axs[1].tick_params(axis='x', labelrotation = 75)
axs[1].xaxis.label.set_visible(False)
axs[1].set_ylim([0.4, 1])

sns.barplot(x="Method", y="Fraction of inliers", data=df_owl, palette=palette, ax=axs[2])
axs[2].tick_params(axis='x', labelrotation = 75)
axs[2].xaxis.label.set_visible(False)
axs[2].set_ylim([0.4, 1])

plt.tight_layout()
plt.savefig("figures/ari_comp.pdf", bbox_inches='tight')

