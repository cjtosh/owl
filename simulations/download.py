from sklearn.datasets import fetch_openml
import urllib.request
import pickle
import os
import shutil
import pandas as pd


os.makedirs('data', exist_ok=True)

mnist = fetch_openml('mnist_784')

with open("data/mnist.pkl", 'rb') as io:
    pickle.dump(mnist, io)


urllib.request.urlretrieve("https://github.com/hoonose/sever/blob/master/svm/data/enron_data.mat?raw=true", "data/enron_data.mat")

os.makedirs("temp", exist_ok=True)
urllib.request.urlretrieve("https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/spwgrcnjdg-1.zip", "temp/all_qsar.zip")


import zipfile
with zipfile.ZipFile("temp/all_qsar.zip","r") as zip_ref:
    zip_ref.extractall("temp/all_qsar")
    
with zipfile.ZipFile("temp/all_qsar/datasets.zip","r") as zip_ref:
    zip_ref.extractall("temp/all_qsar/datasets")
    
df = pd.read_csv("temp/all_qsar/datasets/originals/data_CHEMBL203.csv")
df.to_csv("data/qsar.csv")
shutil.rmtree('temp')


urllib.request.urlretrieve("https://drive.google.com/u/0/uc?id=1E9ye9knR2scbmW9Kp_lLbdq7JQdsyOAe&export=download", "data/cell_data.csv")
