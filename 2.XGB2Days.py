import os
import json
import pandas as pd
import numpy as np

from Models.XGBoost.XGBoost import run_xgboost_different_datasets

def main():
    ##1. read configuration file
    configs = json.load(open('Configuration.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']) : os.makedirs(configs['model']['save_dir'])


    grouping = configs['data']['grouping']

    outcomes = (configs['data']['classification_outcome'])

    ##2. read data
    timeseries_path = configs['paths']['data_path']


    for outcome in outcomes:
        time_series = pd.read_csv(timeseries_path + "SMOTEDTimeSeries/" + outcome + "FlatTimeSeries2Days.csv")
        non_smoted_time_series = pd.read_csv(timeseries_path + "NonSMOTEDTimeSeries/"+outcome+"FlatTimeSeries2Days.csv")

        experiment_number = "2Days"
        fn_x, fn_y, fn_id = run_xgboost_different_datasets(time_series, non_smoted_time_series,
                                       outcome, grouping, experiment_number )

        fn_x = pd.DataFrame(fn_x)
        fn_x[outcome] = fn_y
        fn_x[grouping] = fn_id
        fn_x.to_csv("false_negatives"+outcome+".csv", index=False)
        fn_x = fn_x.drop_duplicates()
        print(fn_x)
if __name__ == '__main__':
    main()
