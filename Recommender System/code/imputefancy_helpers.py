#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from fancyimpute import IterativeSVD, NuclearNormMinimization
from helpers import create_submission
import math


def compute_rmse(data, pred, nnz_indices):            
    rmse = 0
    for row , col in nnz_indices:
        rmse += (data[row,col] - pred[row,col])**2

    rmse = math.sqrt(rmse/(2*len(nnz_indices)))                 
    return rmse

def SVD_predict(data, test, path_to_file, svd_rank = 30):
    data[data==0]= np.nan
    completed_matrix = IterativeSVD(rank = svd_rank).complete(data.todense())
    nnz_row, nnz_column = test.todense().nonzero()
    nnz_test = list(zip(nnz_row, nnz_column))
    create_submission( path_to_file, completed_matrix, nnz_test)

def SVD_NuclearNormMinimization(data, test, path_to_file):
    data[data==0]= np.nan
    completed_matrix = NuclearNormMinimization().complete(data.todense())
    nnz_row, nnz_column = test.todense().nonzero()
    nnz_test = list(zip(nnz_row, nnz_column))
    create_submission( path_to_file, completed_matrix, nnz_test)


def SVD_estimate_error(train, test, svd_rank = 30):
    train[train==0]= np.nan
    completed_matrix = IterativeSVD(rank = svd_rank).complete(train.todense())
    test_dense = test.todense()
    nnz_row, nnz_column = test_dense.nonzero()
    nnz_test = list(zip(nnz_row, nnz_column))
    return compute_rmse(test_dense, completed_matrix, nnz_test)
