import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def save_model(clf, filename):
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)

if __name__ == '__main__':
    combined = pd.read_csv('Combined.csv')

    lof = LocalOutlierFactor()

    X = combined.drop(['Target'], axis=1)
    y = combined['Target']

    pred = []

    for i in range(len(X)):
        inputs = X.values[i].reshape(-1, 1)
        prediction = lof.fit_predict(inputs)
        save_model(lof, 'LOF.pkl')

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

    conf_matrix = confusion_matrix(y, pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.show()

    print('SUCCESS: Model saved to LOF.pkl')