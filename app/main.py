from flask import Flask, request
from flask_cors import CORS, cross_origin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# set verified origin of client
app.config['TARGET_ORIGIN'] = '*'

@app.route('/')
def index():
    return "<p>Server is up and running</p>"

@app.route('/lof_test', methods=['POST'])
@cross_origin(origin=app.config['TARGET_ORIGIN'], headers=['Content-Type'])
def lof_test():
    if request.method == 'POST':
        # fetch received data
        request_data = request.get_json()

        # get the data and the columns
        data = [request_data['consumption']]
        columns = request_data['columns']

        # transform received data to a DataFrame
        normal = pd.DataFrame(data, columns=columns)
        normal = normal.drop(['User'], axis=1)

        # initialize variables to return
        prediction_result = ''
        inlier = []
        outlier = []
        inlierId = []
        outlierId = []

        # local outlier factor
        lof = LocalOutlierFactor()
        inputs = normal.values[0].reshape(-1, 1)
        prediction = lof.fit_predict(inputs)

        # attach prediction result
        if -1 not in prediction:
            prediction_result = 'normal'
        else:
            prediction_result = 'abnormal'

        # attach outliers and inliers
        for i, j in zip(inputs, prediction):
            if j == -1:
                outlier.append(float(i[0]))
            else:
                inlier.append(float(i[0]))
        
        # attach corresponding record IDs of outliers and inliers
        for x, y in zip(columns[1:], prediction):
            if x != 'User':
                if y == -1:
                    outlierId.append(x)
                else:
                    inlierId.append(x)
        
        # return results
        return { "prediction": prediction_result, "inlier": inlier, "outlier": outlier, "inlierId": inlierId, "outlierId": outlierId }, 200