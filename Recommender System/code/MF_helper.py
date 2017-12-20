# -*- coding: utf-8 -*-

"""
Helper functions for the custom matrix_factorization algorithms included in
models.py (matrix_factorization_SGD and matrix_factorization_ALS)
    
Created on Tue Dec 19 2017

@author: Andrea Manzini, Lorenzo Lazzara, Farah Charab
"""

import numpy as np
import scipy.sparse as sp
from itertools import groupby

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]
        
        # update column row of user features
        V = M @ train[user, items].T
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[users, item]
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features

def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data

def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices

def init_MF(train, num_features):
        """ Not a model:
            init the parameter for matrix factorization."""
            
        num_user, num_item = train.get_shape()
        
        user_features = np.random.rand(num_features, num_user)
        item_features = np.random.rand(num_features, num_item)
        
        #start by item features.
        item_nnz = train.getnnz(axis=0)
        item_sum = train.sum(axis=0)
        
        for ind in range(num_item):
            item_features[0, ind] = item_sum[0, ind] / item_nnz[ind]
        
        return user_features, item_features

def compute_error_MF(data, user_features, item_features, nz):
        """compute the loss (MSE) of the prediction of nonzero elements."""
        mse = 0
        for row, col in nz:
            item_info = item_features[:, col]
            user_info = user_features[:, row]
            mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
        return np.sqrt(1.0 * mse / len(nz))