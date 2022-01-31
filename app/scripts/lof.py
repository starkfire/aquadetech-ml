# This is a demo script which takes data from Normal.csv, Abnormal.csv, and Combined.csv
# and runs Local Outlier Factor against the datasets
#
# This demo script will also display:
#    - scatterplots for showing outliers and inliers from the datasets
#    - a confusion matrix which serves as a visual metric

# IN[2]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# Local Outlier Factor
## unsupervised (semi-supervised) anomaly detection machine leaning algorithm

# IN[3] - IN[5]
normal = pd.read_csv('Normal.csv')
normal = normal.drop(['User'], axis=1)

abnormal = pd.read_csv('Abnormal.csv')
abnormal = abnormal.drop(['User'], axis=1)

combined = pd.read_csv('Combined.csv')

# IN[6]
# normal prediction
lof = LocalOutlierFactor()
inputs = normal.values[0].reshape(-1, 1)

prediction = lof.fit_predict(inputs)
if -1 not in prediction:
    print('normal')
else:
    print('abnormal')
    
for i,j in zip(prediction, inputs):
    print(i,j)

# IN[7]
# for graph
outlier = []
inlier = []
for i, j in zip(inputs, prediction):
    if j==-1:
        outlier.append(float(i[0]))
    else:
        inlier.append(float(i[0]))
plt.title('No Outliers')
plt.scatter(inlier, inlier, color='b')
plt.scatter(outlier, outlier, color='r')
plt.show()

# IN[8]
# abnormal prediction
lof = LocalOutlierFactor()
inputs = abnormal.values[0].reshape(-1, 1)

columnList = abnormal.columns.to_list()

prediction = lof.fit_predict(inputs)
if -1 not in prediction:
    print('normal')
else:
    print('abnormal')

# display month and year if result is abnormal
for i in range(len(inputs)):
    if prediction[i]!=-1:
        print(prediction[i], inputs[i])
    else:
        print(prediction[i], inputs[i], columnList[i])

# IN[9]
# for graph
outlier = []
inlier = []
for i, j in zip(inputs, prediction):
    if j==-1:
        outlier.append(float(i[0]))
    else:
        inlier.append(float(i[0]))

plt.title('With Outliers')
plt.scatter(inlier, inlier, color='b')
plt.scatter(outlier, outlier, color='r')
plt.show()

# IN[10]
# getting accuracy and other evaulation metrics
X = combined.drop(['Target'], axis=1)
y = combined['Target']

lof = LocalOutlierFactor()

# getting prediction per input in the dataset
pred = []
for i in range(len(X)):
    inputs = X.values[i].reshape(-1, 1)
    prediction = lof.fit_predict(inputs)
    if -1 not in prediction : 
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

# IN[11]
customInput = [['17.135', '15.972', '25.729', '17.787', '17.286', '21.169',
       '28.685', '21.328', '17.818', '21.667', '17.616', '22.182',
       '27.491', '27.953', '17.645', '26.142', '24.72', '26.346',
       '20.361', '18.388', '24.114', '20.071', '19.106', '22.04',
       '27.666', '28.388', '20.047', '20.655', '22.649', '26.831',
       '15.918', '21.845', '25.001', '29.83', '24.375', '26.378']]

lof = LocalOutlierFactor()
inputs = np.array(customInput).reshape(-1, 1)

start = time.time() # start timer
prediction = lof.fit_predict(inputs)
print('Stop 1 prediction output: ' + str(prediction))
stop1 = time.time() # end timer
if -1 not in prediction:
    print('Stop 2 prediction output: ' + str('Normal'))
else:
    print('Stop 2 prediction: ' + str('Abnormal'))
stop2 = time.time() # end timer
    

print(f'Training and Predict Time1: {stop1-start}s')
print(f'Training and Predict Time2: {stop2-start}s')

# IN[23]
# Training Time
start = time.time() # start timer
prediction = lof.fit(X)
stop = time.time() # end timer
print(f'Training Time: {stop-start}s')

# Confusion Matrix
conf_matrix = confusion_matrix(y, pred)
sns.heatmap(conf_matrix, annot=True,cmap='Blues', fmt='g')
plt.show()