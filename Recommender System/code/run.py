# -*- coding: utf-8 -*-

"""
This file contains the main code of the recommendations system. It load the
datasets from the kaggle csv files and train all the models on them. Then,
it computes the blending of all the models with a ridge regression. Finally,
it combines the predictions given by the models, using the weights found with
the blending and generate the submission for Kaggle.

The parameters of the models are hardcoded in their respective functions. Some
models are custom and entirely written in the functions, while others come from
the Surprise library.

Here we use a custom version of the surprise library, with some improvements.
Efforts were expecially made to optimize SVDpp, which, otherwise, would have
been too slow for the hyperparameters' search.

Each model function creates a dump of the predictions and during the blending
all the predictions are collected again from the files. Default location for
the dump are: ../predictions and ../test

The code takes approximately 4 hours on a Intel i7 6700HQ. The algorithms are
not parallelized, so only a fraction of the total computational power of the
CPU is used. The RAM usage is always lower than 4GB, so it should work
perfectly on a 8GB system.

Created on Tue Dec 19 2017

@author: Andrea Manzini, Lorenzo Lazzara, Farah Charab
"""

import random
import os
from run_helper import (import_kaggle_trainset, import_kaggle_testset,
                             load_all_predictions, blending,
                             generate_submission)
from models import (global_mean, user_mean, item_mean, user_median,
                    item_median, matrix_factorization_SGD,
                    matrix_factorization_ALS, slope_one, svd, knn_user,
                    knn_item, baseline, svdpp)

#%% IMPORT DATA

print('IMPORT DATA...', end='\n\n')

# Define location of kaggle csv files
path_dataset = os.path.join('..','data','data_train.csv')
path_samplesub = os.path.join('..','data','sample_submission.csv')

data = import_kaggle_trainset(path_dataset)

random.seed(10)

# Split data in 15 folds, but only keeps the first trainset
# and the first testset (equivalent to a splitting 93.3%, 6.7%)
data.split(n_folds=15)
# use next to obtain the first element of the generator
trainset, testset = next(data.folds())

predset = import_kaggle_testset(path_samplesub)

#%% GENERATE PREDICTION OF RECOMMENDATION MODELS

print('GENERATE PREDICTIONS...', end='\n\n')

t = time()

global_mean(trainset, testset, predset)
user_mean(trainset, testset, predset)
item_mean(trainset, testset, predset)
user_median(trainset, testset, predset)
item_median(trainset, testset, predset)
matrix_factorization_SGD(trainset, testset, predset)
matrix_factorization_ALS(trainset, testset, predset)
baseline(trainset, testset, predset)
slope_one(trainset, testset, predset)
knn_user(trainset, testset, predset)
knn_item(trainset, testset, predset)
svd(trainset, testset, predset)
svdpp(trainset, testset, predset)

#%% Compute the blending of the models

print('BLENDING...', end='\n\n')

X_test, X_pred, models = load_all_predictions(len(testset), len(predset))
final_predictions, weights = blending(X_test, X_pred, models, testset)

#%% Write final predictions

print('GENERATE SUBMISSION...', end='\n\n')

generate_submission(final_predictions, trainset, predset)
