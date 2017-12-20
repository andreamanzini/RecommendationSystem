# -*- coding: utf-8 -*-

"""
Other helper functions used during the development of the project
    
Created on Tue Dec 19 2017

@author: Andrea Manzini, Lorenzo Lazzara, Farah Charab
"""

import numpy as np
import pandas as pd
import random
from suprise import SVD, accuracy, KNNBaseline, BaselineOnly, print_perf, evaluate
from format_helper import export_prediction
from plots import plot_traintest_over_factors, plot_train_over_factors, plot_test_over_factors
from tqdm import tqdm

def produce_svd_factors_plot(data):
    factors_list = np.arange(10,140,10)
    tot_rmse_tr = []
    tot_rmse_te = []
    for n_factors in factors_list:
        algo = SVD(n_factors=n_factors, n_epochs=30, lr_bu=0.01, lr_bi=0.01, lr_pu=0.1, lr_qi=0.1, reg_bu=0.05, reg_bi=0.05, reg_pu=0.09, reg_qi=0.1)
        rmse_tr = np.zeros(data.n_folds)
        rmse_te = np.zeros(data.n_folds)
        for i, (trainset, testset) in enumerate(tqdm(data.folds())):
        
            # train and test algorithm.
            algo.train(trainset)
            train_pred = algo.test(trainset.build_testset())
            test_pred = algo.test(testset)
            
            rmse_tr[i] = accuracy.rmse(train_pred, verbose=False)
            rmse_te[i] = accuracy.rmse(test_pred, verbose=False)
        
        # Compute and print Root Mean Squared Error
        rmse_tr_mean = np.mean(rmse_tr)
        rmse_te_mean = np.mean(rmse_te)
        print('Factor: ', n_factors)
        print('   Train score: ', rmse_tr_mean)
        print('   Test Score: ', rmse_te_mean)
        
        # Store the rmse on train and test of this n_factors
        tot_rmse_tr.append(rmse_tr_mean)
        tot_rmse_te.append(rmse_te_mean)
        
    # Plot and sasve figures
    plot_traintest_over_factors(factors_list, tot_rmse_tr, tot_rmse_te, '../figures/SVD_over_factors.png')
    plot_train_over_factors(factors_list, tot_rmse_tr, tot_rmse_te, '../figures/SVD_over_factors_train.png')
    plot_test_over_factors(factors_list, tot_rmse_tr, tot_rmse_te, '../figures/SVD_over_factors_test.png')
        
    

def pred_final_BaselineOnly(data, test, bsl_options):
    """ Find prediction for BaselineOnly method using entire trainset
        and export them in the kaggle format"""
    
    trainset = data.build_full_trainset()
    algo = BaselineOnly(bsl_options=bsl_options)
    algo.train(trainset)
    prediction = algo.test(test)
    export_prediction(prediction)
    
def pred_final_KNNBaseline(data, test, k=40, min_k=1, sim_options={}, bsl_options={}):
    """ Find prediction for BaselineOnly method using entire trainset
        and export them in the kaggle format"""
    
    trainset = data.build_full_trainset()
    algo = KNNBaseline(k, min_k, sim_options, bsl_options)
    algo.train(trainset)
    prediction = algo.test(test)
    export_prediction(prediction)

def pred_final_SVD(data, test, n_factors = 100, n_epochs =20, biased = True, 
                       lr_all = 0.005, reg_all = 0.02, 
                       verbose=True):
    """
    Find prediction for SVD using entire method using entire trainset
        and export them in the kaggle format
        
    n_factors – The number of factors. 
    n_epochs – The number of iteration of the SGD procedure. 
    biased (bool) – Whether to use baselines (or biases). 
    lr_all – The learning rate for all parameters. 
    reg_all – The regularization term for all parameters. 
    verbose – If True, prints the current epoch. 
    """
    trainset = data.build_full_trainset()
    algo = SVD(n_factors = n_factors, n_epochs = n_epochs, biased = biased, 
               lr_all=lr_all, reg_all = reg_all, 
                       verbose = verbose)
    algo.train(trainset)
    prediction = algo.test(test)
    export_prediction(prediction)

def SVD_cross_validate(data, n_factors = 100, n_epochs =20, biased = True, 
                       lr_all = 0.005, reg_all = 0.02, 
                       verbose=True, nfolds = 4, measures = ['RMSE']):
    """
    Uses cross validation to compute estimated error on the training set using 
    metrics passed by array measures.
        
    n_factors     – The number of factors. 
    n_epochs      – The number of iteration of the SGD procedure. 
    biased (bool) – Whether to use baselines (or biases). 
    lr_all        – The learning rate for all parameters. 
    reg_all       – The regularization term for all parameters. 
    verbose       – If True, prints the current epoch. 
    nfolds        - number of folfs used in cross validation.  
    measures      - array of measures used to measure the error. 
                  Available metrics: RMSE (Root Mean Squared Error), 
                  MAE (Mean Absolute Error), 
                  FCP (Fraction of Concordant Pairs).
    """
    random.seed(10)
    data.split(nfolds)
    algo = SVD(n_factors = n_factors, n_epochs = n_epochs, biased = biased, 
               lr_all=lr_all, reg_all = reg_all, 
                       verbose = verbose)    
    perf = evaluate(algo, data, measures)
    print_perf(perf)
    
def save_gridsearch_info(grid_search, file_name):
    """ Save gridsearch info in the dataframe format dor further analysis """
    results_df = pd.DataFrame.from_dict(grid_search.cv_results)
    results_df.to_csv('../method_reports/' + file_name)

