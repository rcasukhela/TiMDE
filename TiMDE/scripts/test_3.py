import os
import pickle

def predict():
    l = [0, 1, 2]
    with open('./output_pred/pred.pkl', 'wb') as f:
        pickle.dump(l, f)