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
from werkzeug.utils import secure_filename
import pickle

app = Flask(__name__)

# load an existing model or .pkl file
def load_model(filename):
    with open(filename, 'rb') as f:
        clf = pickle.load(f)
    return clf

# write an object to a .pkl file
def save_model(clf, filename):
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)

# set verified origin of client
app.config['TARGET_ORIGIN'] = '*'

@app.route('/', methods=['GET'])
@cross_origin(origin='*')
def index():
    return "<p>Server is up and running</p>"

@app.route('/lof_train_from_file', methods=['POST'])
@cross_origin(origin=app.config['TARGET_ORIGIN'], headers=['Content-Type'])
def lof_train_from_file():
    if request.method == 'POST':
        f = request.files['file']
        filename_full = f.filename.split('.')
        fileName, fileFormat = filename_full[0], filename_full[-1]
        if fileFormat != 'csv':
            return { 'message': 'Invalid File Format' }, 400
        else:
            try:
                filename = secure_filename(f.filename)
                f.save(filename)
                
                combined = pd.read_csv(filename)

                # load existing model
                lof = load_model('LOF.pkl')

                x = combined.drop(['Target'], axis=1)
                y = combined['Target']

                # initialize variables to return
                predictions = []

                try:
                    for i in range(len(x)):
                        inputs = x.values[i].reshape(-1, 1)
                        prediction = lof.fit_predict(inputs)

                        # update existing model
                        save_model(lof, 'LOF.pkl')

                        if -1 not in prediction:
                            predictions.append('Normal')
                        else:
                            predictions.append('Abnormal')
                except:
                    return { "message": "Your dataset contains null or invalid values" }, 400
                
                n = 0
                a = 0
                na = 0
                an = 0
                score = 0

                for i, j in zip(y, predictions):
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
                
                accuracy = str((score/len(x))*100)

                print('Accuracy: ' + accuracy)
                print('Classified Normal as Normal: '+str(n))
                print('Classified Abnormal as Abnormal: '+str(a))
                print('Classified Normal as Abnormal: '+str(na))
                print('Classified Abnormal as Normal: '+str(an))

                return { "accuracy": accuracy, "true_normal": n, "true_abnormal": a, "false_normal": an, "false_abnormal": na }, 200
            except:
                return { "message": 'An Unknown Error Occurred' }, 400

@app.route('/lof_train', methods=['POST'])
@cross_origin(origin=app.config['TARGET_ORIGIN'], headers=['Content-Type'])
def lof_train():
    if request.method == 'POST':
        # fetch received data
        request_data = request.get_json()

        # get the data and the columns
        data = request_data['normal'] + request_data['abnormal']
        columns = request_data['columns']

        # transform received data to a DataFrame
        combined = pd.DataFrame(data, columns=columns)

        x = combined.drop(['Target'], axis=1)
        y = combined['Target']

        # initialize variables to return
        predictions = []

        # import the existing model
        lof = load_model('LOF.pkl')

        for i in range(len(x)):
            inputs = x.values[i].reshape(-1, 1)
            prediction = lof.fit_predict(inputs)
            
            # update the existing model
            save_model(lof, 'LOF.pkl')

            # gather predictions
            if -1 not in prediction:
                predictions.append('Normal')
            else:
                predictions.append('Abnormal')
        
        # metrics/results
        n = 0
        a = 0
        na = 0
        an = 0
        score = 0

        for i, j in zip(y, predictions):
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
        
        accuracy = str((score/len(x))*100)

        print('Accuracy: ' + accuracy)
        print('Classified Normal as Normal: '+str(n))
        print('Classified Abnormal as Abnormal: '+str(a))
        print('Classified Normal as Abnormal: '+str(na))
        print('Classified Abnormal as Normal: '+str(an))

        return { "accuracy": accuracy, "true_normal": n, "true_abnormal": a, "false_normal": an, "false_abnormal": na }, 200


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

        # start timer
        start = time.time()

        # predict
        prediction = lof.fit_predict(inputs)

        # stop timer
        end = time.time()

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
        return { "prediction": prediction_result, "inlier": inlier, "outlier": outlier, "inlierId": inlierId, "outlierId": outlierId, "trainingTime": end-start }, 200