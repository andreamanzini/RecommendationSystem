from surprise import Dataset, BaselineOnly, evaluate, print_perf
from format_helper import export_predictions

def pred_final_BaselineOnly(data, bsl_options):
    """ Find prediction for BaselineOnly method using entire trainset
        and export them in the kaggle format"""
    
    trainset = data.build_full_trainset()
    algo = BaselineOnly(bsl_options=bsl_options)
    algo.train(trainset)
    prediction = algo.test(trainset.build_anti_testset())
    export_predictions(prediction)

    