# This script takes a custom CSV input, runs the
# Local Outlier Factor against the data, and
# displays a confusion matrix as a visual metric
# of the prediction results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split

import sys

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

try:
    filename = input('Enter filename: ')
    dataset = pd.read_csv(filename)

    lof = LocalOutlierFactor()

    X = dataset.drop(['Target'], axis=1)
    y = dataset['Target']

    pred = []

    for i in range(len(X)):
        inputs = X.values[i].reshape(-1, 1)
        prediction = lof.fit_predict(inputs)
        if -1 not in prediction:
            pred.append('Normal')
        else:
            pred.append('Abnormal')

    n = 0 # normal
    a = 0 # abnormal
    na = 0 # normal classified as abnormal
    an = 0 # abnormal classified as normal
    score = 0 # score for accuracy

    for i, j in zip(y, pred):
        # get accuracy
        if i==j:
            score+=1
        
        # get confusion matrix
        if i==j and i=='Normal':
            n+=1
        elif i==j and i=='Abnormal':
            a+=1
        elif i!=j and i=='Normal':
            na+=1
        elif i!=j and i=='Abnormal':
            an+=1
    
    print('Accuracy: '+str((score/len(X))*100))
    print('Classified Normal as Normal: '+str(n))
    print('Classified Abnormal as Abnormal: '+str(a))
    print('Classified Normal as Abnormal: '+str(na))
    print('Classified Abnormal as Normal: '+str(an))

    # display confusion matrix
    conf_matrix = confusion_matrix(y, pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.show()

except:
    print('Invalid Filename or Dataset')
    sys.exit()