# -*- coding: utf-8 -*-

"""
This file contains all the models used for recommendation. They are divided in:
    custom models:  global_mean, user_mean, item_mean, user_median, item_median
                    matrix_factorization_SGD, matrix_factorization_ALS
    Surprise models: slope_one, baseline, KNN_user, KNN_items, svd, svdpp
    
These are the minutes they take on a Intel i7 6700HQ, 8Gb RAM with data slitted
in 90% training, 10% testing. The models not listed take less than 1 minute.
    matrix_factorization_SGD : 13
    matrix_factorization_ALS : 13
    KNN_user                 : 50
    KNN_item                 : 5
    slope_one                : 4
    svdpp                    : 3
    svdpp                    : 140
    
All the models all perfectly compatible with the trainset and testset format
provided by the Database module of the Surprise library.

SVD and SVDpp models are a slightly modified version of the original Surprise 
algorithms. In order to make them work the best, the modified version of the
Surprise library must be installed.
    
Created on Tue Dec 19 2017

@author: Andrea Manzini, Lorenzo Lazzara, Farah Charab
"""

import numpy as np
import scipy.sparse as sp
from run_helper import save_predictions, is_already_predicted, calculate_rmse
from surprise import SlopeOne, BaselineOnly, KNNBaseline, SVD, SVDpp, accuracy
from MF_helper import (update_user_feature, update_item_feature,
                       build_index_groups, init_MF, compute_error_MF)


def global_mean(trainset, testset, predset):
    """Save predictions based on the global mean"""
    
    modelname = 'globalmean'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return

    print('Global Mean model')
    global_mean = trainset.global_mean
    
    # Find predictions
    train_pred = np.tile(global_mean, trainset.n_ratings)
    test_pred = np.tile(global_mean, len(testset))
    final_pred = np.tile(global_mean, len(predset))
    
    # Extract true labels
    train_labels = [rat for (_,_,rat) in trainset.all_ratings()]
    test_labels = [rat for (_,_,rat) in testset]

    # Evaluate performances
    print('   RMSE on Train: ', calculate_rmse(train_labels, train_pred) )
    rmse = calculate_rmse(test_labels, test_pred)
    print('   RMSE on Test: ', rmse )

    # Save predictions
    save_predictions(modelname, rmse, test_pred, 'test')
    save_predictions(modelname, rmse, final_pred)
    
def user_mean(trainset, testset, predset):
    """Save predictions based on the user means"""
    
    modelname = 'usermean'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    print('User Mean model')

    # Find the mean rating of each user
    user_mean = np.zeros(trainset.n_users + 1) # raw_indices start from 1
    for user in trainset.all_users():
        ratings = [rat for (_,rat) in trainset.ur[user]]
        
        if ratings:
            user_mean[int(trainset.to_raw_uid(user))] = np.mean(ratings)
        else:
            user_mean[int(trainset.to_raw_uid(user))] = trainset.global_mean
            
    # Extract info from datasets
    train_users, _, train_labels = list(zip(*trainset.all_ratings()))
    test_users, _, test_labels = list(zip(*testset))
    pred_users, _, pred_labels = list(zip(*predset))
    
    # Raw ids are strings, so convert them into integer
    train_users = np.array(train_users, dtype='int')
    test_users = np.array(test_users, dtype='int')
    pred_users = np.array(pred_users, dtype='int')
    
    # Calculate predictions
    train_pred = np.empty(trainset.n_ratings)
    test_pred = np.empty(len(testset))
    final_pred = np.empty(len(predset))
    for n, user in enumerate(train_users):
        train_pred[n] = user_mean[int(trainset.to_raw_uid(user))]
    for n, user in enumerate(test_users):
        test_pred[n] = user_mean[user]
    for n, user in enumerate(pred_users):
        final_pred[n] = user_mean[user]
        
    # Evaluate performances
    print('   RMSE on Train: ', calculate_rmse(train_labels, train_pred) )
    rmse = calculate_rmse(test_labels, test_pred)
    print('   RMSE on Test: ', rmse)
    
    # Save predictions
    save_predictions(modelname, rmse, test_pred, 'test')
    save_predictions(modelname, rmse, final_pred)
    
def user_median(trainset, testset, predset):
    """Save predictions based on the user medians"""
    
    modelname = 'usermedian'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    print('User Median model')

    # Find the mean rating of each user
    user_median = np.zeros(trainset.n_users + 1) # raw_indices start from 1
    for user in trainset.all_users():
        ratings = [rat for (_,rat) in trainset.ur[user]]
        
        if ratings:
            user_median[int(trainset.to_raw_uid(user))] = np.median(ratings)
        else:
            user_median[int(trainset.to_raw_uid(user))] = trainset.global_mean
            
    # Extract info from datasets
    train_users, _, train_labels = list(zip(*trainset.all_ratings()))
    test_users, _, test_labels = list(zip(*testset))
    pred_users, _, pred_labels = list(zip(*predset))
    
    # Raw ids are strings, so convert them into integer
    train_users = np.array(train_users, dtype='int')
    test_users = np.array(test_users, dtype='int')
    pred_users = np.array(pred_users, dtype='int')
    
    # Calculate predictions
    train_pred = np.empty(trainset.n_ratings)
    test_pred = np.empty(len(testset))
    final_pred = np.empty(len(predset))
    for n, user in enumerate(train_users):
        train_pred[n] = user_median[int(trainset.to_raw_uid(user))]
    for n, user in enumerate(test_users):
        test_pred[n] = user_median[user]
    for n, user in enumerate(pred_users):
        final_pred[n] = user_median[user]
        
    # Evaluate performances
    print('   RMSE on Train: ', calculate_rmse(train_labels, train_pred) )
    rmse = calculate_rmse(test_labels, test_pred)
    print('   RMSE on Test: ', rmse)
    
    # Save predictions
    save_predictions(modelname, rmse, test_pred, 'test')
    save_predictions(modelname, rmse, final_pred)
    
def item_mean(trainset, testset, predset):
    """Save predictions based on the items means"""
    
    modelname = 'itemmean'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    print('Item Mean model')

    # Find the mean rating of each item
    item_mean = np.zeros(trainset.n_items + 1) # raw_indices start from 1
    for item in trainset.all_items():
        ratings = [rat for (_,rat) in trainset.ir[item]]
        
        if ratings:
            item_mean[int(trainset.to_raw_iid(item))] = np.mean(ratings)
        else:
            item_mean[int(trainset.to_raw_iid(item))] = trainset.global_mean
            
    # Extract info from datasets
    _, train_items, train_labels = list(zip(*trainset.all_ratings()))
    _, test_items, test_labels = list(zip(*testset))
    _, pred_items, pred_labels = list(zip(*predset))
    
    # Raw ids are strings, so convert them into integer
    train_items = np.array(train_items, dtype='int')
    test_items = np.array(test_items, dtype='int')
    pred_items = np.array(pred_items, dtype='int')
    
    # Calculate predictions
    train_pred = np.empty(trainset.n_ratings)
    test_pred = np.empty(len(testset))
    final_pred = np.empty(len(predset))
    for n, item in enumerate(train_items):
        train_pred[n] = item_mean[int(trainset.to_raw_iid(item))]
    for n, item in enumerate(test_items):
        test_pred[n] = item_mean[item]
    for n, item in enumerate(pred_items):
        final_pred[n] = item_mean[item]
        
    # Evaluate performances
    print('   RMSE on Train: ', calculate_rmse(train_labels, train_pred) )
    rmse = calculate_rmse(test_labels, test_pred)
    print('   RMSE on Test: ', rmse)
    
    # Save predictions
    save_predictions(modelname, rmse, test_pred, 'test')
    save_predictions(modelname, rmse, final_pred)
    
def item_median(trainset, testset, predset):
    """Save predictions based on the items medians"""
    
    modelname = 'itemmedian'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    print('Item Median model')

    # Find the mean rating of each item
    item_median = np.zeros(trainset.n_items + 1) # raw_indices start from 1
    for item in trainset.all_items():
        ratings = [rat for (_,rat) in trainset.ir[item]]
        
        if ratings:
            item_median[int(trainset.to_raw_iid(item))] = np.median(ratings)
        else:
            item_median[int(trainset.to_raw_iid(item))] = trainset.global_mean
            
    # Extract info from datasets
    _, train_items, train_labels = list(zip(*trainset.all_ratings()))
    _, test_items, test_labels = list(zip(*testset))
    _, pred_items, pred_labels = list(zip(*predset))
    
    # Raw ids are strings, so convert them into integer
    train_items = np.array(train_items, dtype='int')
    test_items = np.array(test_items, dtype='int')
    pred_items = np.array(pred_items, dtype='int')
    
    # Calculate predictions
    train_pred = np.empty(trainset.n_ratings)
    test_pred = np.empty(len(testset))
    final_pred = np.empty(len(predset))
    for n, item in enumerate(train_items):
        train_pred[n] = item_median[int(trainset.to_raw_iid(item))]
    for n, item in enumerate(test_items):
        test_pred[n] = item_median[item]
    for n, item in enumerate(pred_items):
        final_pred[n] = item_median[item]
        
    # Evaluate performances
    print('   RMSE on Train: ', calculate_rmse(train_labels, train_pred) )
    rmse = calculate_rmse(test_labels, test_pred)
    print('   RMSE on Test: ', rmse)
    
    # Save predictions
    save_predictions(modelname, rmse, test_pred, 'test')
    save_predictions(modelname, rmse, final_pred)

    
def matrix_factorization_SGD(trainset, testset, predset):
    """Save predictions based on the matrix factorization with SGD"""
    
    modelname = 'mfsgd'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    # HARDCODED PARAMETERS
    gamma = 0.06
    num_features = 20   # K in the lecture notes
    lambda_user = 0.08
    lambda_item = 0.1
    num_epochs = 30     # number of full passes through the train set
    
    # Build matrix of train and test for compatibility with this algorithm
    train = sp.lil_matrix((10000, 1000))
    test = sp.lil_matrix((10000, 1000))
    
    # Extract raw_users from trainset
    train_users = [trainset.to_raw_uid(u) for (u,_,_) in trainset.all_ratings()]
    train_items = [trainset.to_raw_iid(i) for (_,i,_) in trainset.all_ratings()]
    train_labels = [r for (_,_,r) in trainset.all_ratings()]
    
    # Extract info from testset and predset
    test_users, test_items, test_labels = list(zip(*testset))
    pred_users, pred_items, pred_labels = list(zip(*predset))
    
    # Raw ids are strings, so convert them into integer.
    # Decrease by 1 because raw ids start from 1 and matrix indices from 0
    train_users = np.array(train_users, dtype='int') - 1
    test_users = np.array(test_users, dtype='int') - 1
    pred_users = np.array(pred_users, dtype='int') - 1
    train_items = np.array(train_items, dtype='int') - 1
    test_items = np.array(test_items, dtype='int') - 1
    pred_items = np.array(pred_items, dtype='int') - 1
    
    # Fill train and test matrices
    train[train_users, train_items] = train_labels
    test[test_users, test_items] = test_labels
    
    # set seed
    np.random.seed(988)
    
    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    print("Matrix Factorization SGD Model")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, n]
            user_info = user_features[:, d]
            err = train[d, n] - user_info.T.dot(item_info)
    
            # calculate the gradient and update
            item_features[:, n] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, d] += gamma * (err * item_info - lambda_user * user_info)
    
    # evaluate the test error
    print('   RMSE on train: ', compute_error_MF(train, user_features, item_features, nz_train))
    rmse = compute_error_MF(test, user_features, item_features, nz_test)
    print("   RMSE on Test: ",rmse)
    
    predictions = user_features.T @ item_features
    save_predictions(modelname, np.asscalar(rmse), predictions[pred_users, pred_items])
    save_predictions(modelname, np.asscalar(rmse), predictions[test.nonzero()], 'test')
    
    
def matrix_factorization_ALS(trainset, testset, predset, verbose=False):
    """ Save predictions based on matrix factorization with ALS"""
    
    modelname = 'mfals'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    # HARDCODED PARAMETERS
    num_features = 20   # K in the lecture notes
    lambda_user = 0.08
    lambda_item = 0.1
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    
    # Build matrix of train and test for compatibility with this algorithm
    train = sp.lil_matrix((10000, 1000))
    test = sp.lil_matrix((10000, 1000))
    
    # Extract raw_users from trainset
    train_users = [trainset.to_raw_uid(u) for (u,_,_) in trainset.all_ratings()]
    train_items = [trainset.to_raw_iid(i) for (_,i,_) in trainset.all_ratings()]
    train_labels = [r for (_,_,r) in trainset.all_ratings()]
    
    # Extract info from testset and predset
    test_users, test_items, test_labels = list(zip(*testset))
    pred_users, pred_items, pred_labels = list(zip(*predset))
    
    # Raw ids are strings, so convert them into integer.
    # Decrease by 1 because raw ids start from 1
    train_users = np.array(train_users, dtype='int') - 1
    test_users = np.array(test_users, dtype='int') - 1
    pred_users = np.array(pred_users, dtype='int') - 1
    train_items = np.array(train_items, dtype='int') - 1
    test_items = np.array(test_items, dtype='int') - 1
    pred_items = np.array(pred_items, dtype='int') - 1
    
    # Fill train and test matrices
    train[train_users, train_items] = train_labels
    test[test_users, test_items] = test_labels
    
    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=1), train.getnnz(axis=0)
    
    # group the indices by row or column index
    nz_train, nz_user_itemindices, nz_item_userindices = build_index_groups(train)

    # run ALS
    print("Matrix Factorization ALS Model")
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error_MF(train, user_features, item_features, nz_train)
        if verbose:
            print("RMSE on Train: {}.".format(error))
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

    # Evaluate train error
    print("RMSE on Train: ", compute_error_MF(train, user_features, item_features, nz_train))

    # evaluate the test error
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    rmse = compute_error_MF(test, user_features, item_features, nnz_test)
    print("RMSE on Test: {v}.".format(v=rmse))
    
    # Save predictions
    predictions = user_features.T @ item_features
    save_predictions(modelname, np.asscalar(rmse), predictions[pred_users, pred_items])
    save_predictions(modelname, np.asscalar(rmse), predictions[test.nonzero()], 'test')
    

def slope_one(trainset, testset, predset):
    
    modelname = 'slopeone'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    algo = SlopeOne()
    print('SlopeOne Model')
    algo.train(trainset)
    
    predictions = algo.test(trainset.build_testset())
    print('   RMSE on Train: ', accuracy.rmse(predictions))
    
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print('   RMSE on Test: ', rmse)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds, 'test')

    print('  Evaluate predicted ratings...')
    predictions = algo.test(predset)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds)
    
def svd(trainset, testset, predset):

    modelname = 'svd'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return

    algo = SVD(n_factors=100, n_epochs=40, lr_bu=0.01, lr_bi=0.01, lr_pu=0.1, lr_qi=0.1, reg_bu=0.05, reg_bi=0.05, reg_pu=0.09, reg_qi=0.1)
    print('SVD Model')
    algo.train(trainset)
    
    predictions = algo.test(trainset.build_testset())
    print('   RMSE on Train: ', accuracy.rmse(predictions))
    
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print('   RMSE on Test: ', rmse)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds, 'test')

    print('   Evaluate predicted ratings...')
    predictions = algo.test(predset)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds)
    

def baseline(trainset, testset, predset):
    
    modelname = 'baseline'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    bsl_options = { 'method': 'als',
                    'reg_i': 1.e-5,
                    'reg_u': 14.6,
                    'n_epochs': 10
                   }
    
    algo = BaselineOnly(bsl_options=bsl_options)
    print('Baseline Model')
    algo.train(trainset)
    
    predictions = algo.test(trainset.build_testset())
    print('   RMSE on Train: ', accuracy.rmse(predictions))
    
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print('   RMSE on Test: ', rmse)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds, 'test')

    print('   Evaluate predicted ratings...')
    predictions = algo.test(predset)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds)
    
def knn_user(trainset, testset, predset):
    
    modelname = 'knnuser'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    bsl_options = { 'method': 'als',
                    'reg_i': 1.e-5,
                    'reg_u': 14.6,
                    'n_epochs': 10
                   }
    sim_options = {
                    'name': 'pearson_baseline',
                    'shrinkage': 100,
                    'user_based': True
                    }
    algo = KNNBaseline(k=300, min_k=10, sim_options=sim_options, bsl_options=bsl_options)
    print('KNN user based Model')
    algo.train(trainset)
    
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print('   RMSE on Test: ', rmse)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds, 'test')

    print('   Evaluate predicted ratings...')
    predictions = algo.test(predset)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds)
    

def knn_item(trainset, testset, predset):
    
    modelname = 'knnitem'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    bsl_options = { 'method': 'als',
                    'reg_i': 1.e-5,
                    'reg_u': 14.6,
                    'n_epochs': 10
                   }
    sim_options = {
                    'name': 'pearson_baseline',
                    'shrinkage': 100,
                    'user_based': False
                    }
    algo = KNNBaseline(k=60, sim_options=sim_options, bsl_options=bsl_options)
    print('KNN item based Model')
    algo.train(trainset)
    
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print('   RMSE on Test: ', rmse)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds, 'test')

    print('   Evaluate predicted ratings...')
    predictions = algo.test(predset)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds)
    
    
def svdpp(trainset, testset, predset):
    
    modelname = 'svdpp'
    # Check if predictions already exist
    if is_already_predicted(modelname):
        return
    
    bsl_options = { 'method': 'als',
                    'reg_i': 1.e-5,
                    'reg_u': 14.6,
                    'n_epochs': 10
                   }
    
    algo = SVDpp(n_epochs=40, n_factors=100, bsl_options=bsl_options, lr_bu=0.01, lr_bi=0.01, lr_pu=0.1, lr_qi=0.1, lr_yj=0.01, reg_bu = 0.05, reg_bi = 0.05, reg_pu = 0.09, reg_qi = 0.1, reg_yj=0.01)
    print('SVDpp Model')
    algo.train(trainset)
    
    predictions = algo.test(trainset.build_testset())
    print('   RMSE on Train: ', accuracy.rmse(predictions))
    
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print('   RMSE on Test: ', rmse)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds, 'test')

    print('   Evaluate predicted ratings...')
    predictions = algo.test(predset)
    preds = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds[j] = pred.est
    save_predictions(modelname, rmse, preds)
    