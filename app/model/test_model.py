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

def load_model(filename):
    with open(filename, 'rb') as f:
        clf = pickle.load(f)
    return clf

if __name__ == '__main__':
    # modify this array with your custom values
    custom_input = [11.886,11.99,12.22,3.063,5.351,4.374,10.737,11.451,11.684,12.37,10.536,12.063,14.87,13.443,12.661,13.649,12.817,14.942,15.257,14.293,13.484,15.048,14.849,13.572,9.127,10.269,9.741,8.7,9.572,8.806,10.296,9.843,9.535,10.295,8.82,8.836]

    print('Reading LOF.pkl')
    model = load_model('LOF.pkl')
    
    print('Testing with Custom Data: ' + str(custom_input))
    data = np.array([custom_input]).reshape(-1, 1)

    print ('Predicting...')
    results = model.fit_predict(data)

    outlier = []
    inlier = []
    for i, j in zip(data, results):
        if j == -1:
            outlier.append(float(i[0]))
        else:
            inlier.append(float(i[0]))
    
    plt.title('Inliers and Outliers')
    plt.scatter(inlier, inlier, color='b')
    plt.scatter(outlier, outlier, color='r')
    plt.show()

    print('Done.')