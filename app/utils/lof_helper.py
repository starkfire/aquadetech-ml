import pandas as pd

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