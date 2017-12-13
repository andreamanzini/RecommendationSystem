# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy
import scipy.sparse as sp
from surprise import Reader, KNNBaseline, BaselineOnly, Dataset, Trainset, SVD, SVDpp, GridSearch, evaluate, accuracy, print_perf, SlopeOne
import random
import itertools
import pandas as pd
from format_helper import *
from surprise_helper import *

# path to dataset file
file_path = ('../data/train_formatted.csv')

# As we're loading a custom dataset, we need to define a reader.
reader = Reader(line_format='user item rating', sep=',')

data = Dataset.load_from_file(file_path, reader=reader)
random.seed(10)
data.split(n_folds=5)  # data can now be used normally


#%%
"""
# Set Grid Parameters
param_grid = {'n_epochs':[10], 'lr_all':list(np.linspace(0.001, 0.002, num=5)), 'reg_all':list(np.linspace(0.05, 0.08, num=5))}

grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])

# Evaluate performances of our algorithm on the dataset.
grid_search.evaluate(data)

# Use Panda dataframe to analyze solution
results_df = pd.DataFrame.from_dict(grid_search.cv_results)
results_df.to_csv('SVD.csv')
print(results_df)
"""
#%%

bsl_options = { 'method': 'als',
                'reg_i': 1.e-5,
                'reg_u': 14.6,
                'n_epochs': 10
               }

#algo = SlopeOne()
#algo = BaselineOnly(bsl_options=bsl_options)
#algo = SVD(n_epochs=30, lr_all=0.0017, reg_all=0.05)
#algo = SVD(n_factors=100, n_epochs=30, lr_bu=0.00267, lr_bi=0.000488, lr_pu=0.01, lr_qi=0.01, reg_all = 0.08)
algo = SVDpp(n_epochs=40, n_factors=50, lr_bu=0.007, lr_bi=0.007, lr_pu=0.007, lr_qi=0.007, lr_yj=0.006, reg_bu=0.01, reg_bi=0.01, reg_pu=0.06, reg_qi=0.06, reg_yj=0.06)
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
test = import_kaggle_testset('../data/sample_formatted.csv')
trainset = data.build_full_trainset()
algo = SVD(n_factors=100, n_epochs=30, lr_bu=0.00267, lr_bi=0.000488, lr_pu=0.01, lr_qi=0.01, reg_all = 0.08)
algo.train(trainset)
prediction = algo.test(test)
export_prediction(prediction)

RMSE: 0.8739
RMSE: 0.9837
RMSE: 0.8725
RMSE: 0.9851
RMSE: 0.8728
RMSE: 0.9867
RMSE: 0.8726
RMSE: 0.9848
RMSE: 0.8719
RMSE: 0.9876
Train score:  0.872744179113
Test Score:  0.985597762733
"""