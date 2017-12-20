# -*- coding: utf-8 -*-

"""
Helper functions for run.py
    
Created on Tue Dec 19 2017

@author: Andrea Manzini, Lorenzo Lazzara, Farah Charab
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.linear_model import Ridge
from glob import glob
from surprise import Dataset, Reader

def import_kaggle_testset(path_samplesub):
    """ Import the kaggle testset from the csv file with the surprise library 
        and return a list of ratings. """
    
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return (row.strip(), col.strip(), rating.strip())

    # read csv file
    fp = open(path_samplesub, "r")
    lines = fp.read().splitlines()[1:]
    fp.close()
    
    # Build a tuple of tuples, one per rating
    ratings = (deal_line(line) for line in lines)
    # Use a Dataframe of pandas only as a mean
    dataframe = pd.DataFrame(ratings, columns=['row','col','rat'])
    # Import ratings in surprise lib    
    data = Dataset.load_from_df(dataframe, Reader())
    # Return the entire testset
    return data.build_full_trainset().build_testset()

def import_kaggle_trainset(path_dataset):
    """ Import the kaggle trainset from the csv file with the surprise library 
        and return a Dataset object. """
    
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return (row.strip(), col.strip(), rating.strip())

    # read csv file
    fp = open(path_dataset, "r")
    lines = fp.read().splitlines()[1:]
    fp.close()
    
    # Build a tuple of tuples, one per rating
    ratings = (deal_line(line) for line in lines)
    # Use a Dataframe of pandas only as a mean
    dataframe = pd.DataFrame(ratings, columns=['row','col','rat'])
    # Import ratings in surprise lib    
    data = Dataset.load_from_df(dataframe, Reader())
    
    return data

def save_predictions(modelname, rmse, pred, basepath='pred'):
    if basepath == 'test':
        basepath = os.path.join('..','test')
    else:
        basepath = os.path.join('..','predictions')
        
    # Create directory if it does not exist
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    
    # Save prediction as pickle serialized file
    paths = os.path.join(basepath, modelname + '_{:.4f}'.format(rmse) + '.pickle')
    fs = open(paths, 'wb')
    pickle.dump(pred, fs)
    fs.close()
    
def is_already_predicted(modelname):
    basepath = os.path.join('..','predictions')
    if glob(os.path.join(basepath, modelname + '_*')):
        return True
    else:
        return False
    
def calculate_rmse(real_labels, predictions):
    """Calculate RMSE."""
    return np.linalg.norm(real_labels - predictions) / np.sqrt(len(real_labels))
    
def load_predictions_from_folder(folderpath, N):
    """Store the predictions exported by the single methods in a matrix"""
    
    # Create a list of valid files
    files = [f for f in os.listdir(folderpath) if f.endswith('.pickle')]
    
    # Initialize matrices for linear regression
    X = np.zeros((len(files), N))
    
    # Iterate over the files in the folder
    models = []
    for i, f in enumerate(files):
        # Create a list of model names
        models.append(f.split('_')[0])

        # Load data from the file and fill the matrix
        path = os.path.join(folderpath, f)
        fp = open(path, 'rb')
        X[i,:] = pickle.load(fp)
        fp.close()
        
    return X, models

def load_all_predictions(N_test, N_pred, basepath='..'):
    """ Load both test and final predictions and store them in two matrices.
        Return also the list of models in the order they compare in the matrices"""
    
    test_path = os.path.join(basepath, 'test')
    pred_path = os.path.join(basepath, 'predictions')
    
    X_test, models_test = load_predictions_from_folder(test_path, N_test)
    X_pred, models_pred = load_predictions_from_folder(pred_path, N_pred)
    
    if models_test != models_pred:
        raise Exception('Models of test and final predictions are different')
        
    print('Loaded {} models'.format(len(models_pred)))
        
    return X_test, X_pred, models_pred

def generate_submission(predicted_ratings, trainset, predset):
    """ Export prediction to kaggle format"""
    # Store users, items and ratings in three arrays
    
    header = 'Id,Prediction\n'
    
    N = len(predicted_ratings)
    users = [int(u) for (u,_,_) in predset]
    items = [int(i) for (_,i,_) in predset]
    rat = predicted_ratings
        
    # Format preditions in the kaggle format
    data = []
    data.append(header) # Add header at the start of the text file
    for j in range(N):
        data.append('r{u}_c{i},{r}\n'.format(u=users[j], i=items[j], r = rat[j]))
        
    # Write predictions in a csv file
    fp = open('./final_prediction.csv', 'w')
    fp.writelines(data)
    fp.close()

def blending(X_test, X_pred, models, testset):
    """ Through a Linear Regression on the test dataset, find the best
        weight for each model and predict the final ratings """

    # Find real labels
    y_test = [rat for (_,_,rat) in testset]

    # Find blending weights
    linreg = Ridge(alpha=0.1, fit_intercept=False)
    linreg.fit(X_test.T, y_test)

    # Create dictionary of weights
    weights = dict(zip(models, linreg.coef_))

    # Predict final ratings
    final_predictions = np.clip(linreg.predict(X_pred.T), 1, 5)

    print('Blending Weights: ')
    print(weights, end='\n\n')
    print('RMSE on Test: %f' % calculate_rmse(y_test, linreg.predict(X_test.T)))

    return final_predictions, weights