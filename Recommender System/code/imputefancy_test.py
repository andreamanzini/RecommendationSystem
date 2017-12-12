# -*- coding: utf-8 -*-
from imputefancy_helpers import *
from helpers import load_data, split_data
from plots import plot_raw_data
import numpy as np

#%%


#%%
path_dataset = '../data/data_train.csv'
ratings= load_data(path_dataset)
num_items_per_user, num_users_per_item = plot_raw_data(ratings)
print("min # of items per user = {}, min # of users per item = {}.".format(
        min(num_items_per_user), min(num_users_per_item)))

#%%

#%%
valid_ratings, train, test = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=0, p_test=0.1)
#%%

SVD_estimate_error(train, test)
#%%

#%%
path_testset = '../data/sample_submission.csv'
test  = load_data(path_testset)
SVD_predict(ratings, test, '../data/final_submission.csv')


SVD_NuclearNormMinimization(ratings, test, '../data/final_submission.csv')

#%%
nnz_row, nnz_column = test.todense().nonzero()
nnz_test = list(zip(nnz_row, nnz_column))
create_submission( '../data/final_submission.csv', completed_matrix, nnz_test)