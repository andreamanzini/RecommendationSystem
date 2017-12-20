"""
Template for testing the Surprise models
    
Created on Tue Dec 19 2017

@author: Andrea Manzini, Lorenzo Lazzara, Farah Charab
"""

import numpy as np
import os
import random
from surprise import KNNBaseline, GridSearch, accuracy
from run_helper import import_kaggle_trainset, import_kaggle_testset
from other_helpers import pred_final_KNNBaseline
from tqdm import tqdm

#%% IMPORT DATA

print('IMPORT DATA...', end='\n\n')

# Define location of kaggle csv files
path_dataset = os.path.join('..','data','data_train.csv')
path_samplesub = os.path.join('..','data','sample_submission.csv')

data = import_kaggle_trainset(path_dataset)

random.seed(10)

# Split data in 5 folds for cross validation
data.split(n_folds=55)

#%% GENERAL PARAMETERS SETTING

bsl_options = { 'method': 'als',
                'reg_i': 1.e-5,
                'reg_u': 14.6,
                'n_epochs': 10
               }

sim_options = { 'name': 'pearson_baseline',
               'shrinkage': 50,
               'user_based': False,
               'min_support': 30
               }

k = 100
min_k = 10



#%% GRID SEARCH OF PARAMETERS
"""
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
"""
#%% EVALUATE SINGLE ALGORITHM
"""
algo = KNNBaseline(k=k, min_k=min_k, sim_options=sim_options, bsl_options=bsl_options)
rmse_tr = np.zeros(data.n_folds)
rmse_te = np.zeros(data.n_folds)
i=0
for i, (trainset, testset) in enumerate(tqdm(data.folds())):

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
"""
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
"""

