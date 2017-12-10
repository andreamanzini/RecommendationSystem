from surprise import Dataset, BaselineOnly, evaluate, print_perf, Reader
from format_helper import export_prediction

def pred_final_BaselineOnly(data, test, bsl_options):
    """ Find prediction for BaselineOnly method using entire trainset
        and export them in the kaggle format"""
    
    trainset = data.build_full_trainset()
    algo = BaselineOnly(bsl_options=bsl_options)
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