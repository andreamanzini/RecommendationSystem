import numpy as np
import scipy
import scipy.sparse as sp
from surprise import Reader, KNNBaseline, BaselineOnly, Dataset, Trainset, SVD, GridSearch, evaluate, accuracy, print_perf
import random
import itertools
import pandas as pd
from surprise_helper import *

# path to dataset file
file_path = ('../data/train_formatted.csv')

# As we're loading a custom dataset, we need to define a reader.
reader = Reader(line_format='user item rating', sep=',')

data = Dataset.load_from_file(file_path, reader=reader)
random.seed(10)
data.split(n_folds=4)  # data can now be used normally

#%% GENERAL PARAMETERS SETTING

bsl_options = { 'method': 'als',
                'reg_i': 1.e-5,
                'reg_u': 14.6,
                'n_epochs': 10
               }

#%% GRID SEARCH OF PARAMETERS

# Set Grid Parameters
param_grid = {
        'bsl_options': { 
                'method': ['als'],
                'reg_i': [1.e-5],
                'reg_u': [14.6],
                'n_epochs': [5]
                },
        
        'sim_options': {
                'name': ['pearson_baseline'],
                'shrinkage': [100]
                },
        
        'k' : [300]
}

# Init grid_search
grid_search = GridSearch(KNNBaseline, param_grid, measures=['RMSE'])

# Evaluate performances of our algorithm on the dataset.
grid_search.evaluate(data)

# Print best score and best parameters
print('Best Score: ', grid_search.best_score)
print('Best parameters: ', grid_search.best_params)

#%% EVALUATE SINGLE ALGORITHM
"""
algo = BaselineOnly(bsl_options=bsl_options)
rmse_tr = np.zeros(data.n_folds)
rmse_te = np.zeros(data.n_folds)
i=0
for trainset, testset in data.folds():

    # train and test algorithm.
    algo.train(trainset)
    train_pred = algo.test(trainset.build_testset())
    test_pred = algo.test(testset)
    
    rmse_tr[i] = accuracy.rmse(train_pred)
    rmse_te[i] = accuracy.rmse(test_pred)
    
    i=i+1

# Compute and print Root Mean Squared Error
print('Train score: ', np.mean(rmse_tr))
print('Test Score: ', np.mean(rmse_te))
"""
#%% PREDICT AND WRITE KAGGLE PREDICTIONS

bsl_options = { 'method': 'als',
                'reg_i': 1.e-5,
                'reg_u': 14.6,
                'n_epochs': 10
               }
sim_options = {
                'name': 'pearson_baseline',
                'shrinkage': 100
                }
k = 300
min_k = 10

test = import_kaggle_testset('../data/sample_formatted.csv')
pred_final_KNNBaseline(data, test, k, min_k, sim_options, bsl_options)