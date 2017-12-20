# -*- coding: utf-8 -*-

import pandas as pd
import random
from suprise import *
from format_helper import *


"""
Other helper functions used during the development of the project
    
Created on Tue Dec 19 2017

@author: Andrea Manzini, Lorenzo Lazzara, Farah Charab
"""

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

