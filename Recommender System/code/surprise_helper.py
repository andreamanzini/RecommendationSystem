from surprise import Dataset, KNNBaseline, BaselineOnly, evaluate, print_perf, Reader
from format_helper import export_prediction
import pandas as pd

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

    
def import_kaggle_testset(path_samplesub):
    # Read sample submission from kaggle
    reader = Reader(line_format='user item rating', sep=',')
    test_data = Dataset.load_from_file(path_samplesub, reader)
    
    # Build testset based on the ratings in the sample
    test = test_data.build_full_trainset().build_testset()
    return test

def save_gridsearch_info(grid_search, file_name):
    # Use Panda dataframe to analyze solution
    results_df = pd.DataFrame.from_dict(grid_search.cv_results)
    results_df.to_csv('../method_reports/' + file_name)