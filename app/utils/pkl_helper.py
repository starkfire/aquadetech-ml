import pickle

# load an existing model or .pkl file
def load_model(filename):
    with open(filename, 'rb') as f:
        clf = pickle.load(f)
    return clf

# write an object to a .pkl file
def save_model(clf, filename):
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)