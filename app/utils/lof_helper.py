import pandas as pd

# classifies data points from the results whether they are outliers or inliers
def classify_data_points(inputs, predictions):
    outlier = []
    inlier = []

    for i, j in zip(inputs, predictions):
        if j == -1:
            outlier.append(float(i[0]))
        else:
            inlier.append(float(i[0]))
    
    return { "outlier": outlier, "inlier": inlier }


# method for getting the database IDs of inliers and outliers
def get_data_point_ids(columns, predictions):
    inlierId = []
    outlierId = []

    for x, y in zip(columns[1:], predictions):
        if x != 'User':
            if y == -1:
                outlierId.append(x)
            else:
                inlierId.append(x)

    return { "outlierId": outlierId, "inlierId": inlierId }


# checks if outliers exist within predictions
# this method is only compatible with lof_test,
# which only runs LOF against a single row of data
def get_prediction_result(predictions):
    if -1 not in predictions:
        return 'normal'
    else:
        return 'abnormal'


# summarizes the results of a LOF analysis
def get_metrics(data, truth, predictions):
    n = 0
    a = 0
    na = 0
    an = 0
    score = 0

    for i, j in zip(truth, predictions):
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
    
    accuracy = (score / len(data)) * 100

    print('Accuracy: ' + str(accuracy))
    print('Classified Normal as Normal: '+str(n))
    print('Classified Abnormal as Abnormal: '+str(a))
    print('Classified Normal as Abnormal: '+str(na))
    print('Classified Abnormal as Normal: '+str(an))

    return { "accuracy": accuracy, "true_normal": n, "true_abnormal": a, "false_normal": an, "false_abnormal": na }