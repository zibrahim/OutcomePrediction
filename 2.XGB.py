import os
import json
import pandas as pd
import numpy as np

import xgboost as xgb
from Models.XGBoost.XGBoost import XGBoostClassifier

def main():
    ##1. read configuration file
    configs = json.load(open('Configuration.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']) : os.makedirs(configs['model']['save_dir'])


    grouping = configs['data']['grouping']

    outcomes = (configs['data']['classification_outcome'])

    ##2. read data
    timeseries_path = configs['paths']['data_path']


    for outcome in outcomes:
        time_series = pd.read_csv(timeseries_path + "SMOTEDTimeSeries/" + outcome + "FlatTimeSeries1Day.csv")
        non_smoted_time_series = pd.read_csv(timeseries_path + "NonSMOTEDTimeSeries/"+outcome+"FlatTimeSeries1Day.csv")

        experiment_number = "1DaySMOTE"
        xgbc = XGBoostClassifier(time_series, non_smoted_time_series,outcome, grouping)
        fn_x, fn_y, fn_id = xgbc.run_xgb(experiment_number, True)

        fn_x = pd.DataFrame(fn_x)
        fn_x[outcome] = fn_y
        fn_x[grouping] = fn_id
        fn_x = fn_x.drop_duplicates(subset=[grouping])
        fn_x.to_csv("false_negativesSMOTE"+outcome+".csv", index=False)

        xgbc.get_feature_importance(experiment_number)


        ##NO SMOTE
        experiment_number = "1DayNoSMOTE"
        fn_x, fn_y, fn_id = xgbc.run_xgb(experiment_number, False)

        fn_x = pd.DataFrame(fn_x)
        fn_x[outcome] = fn_y
        fn_x[grouping] = fn_id
        fn_x = fn_x.drop_duplicates(subset=[grouping])
        fn_x.to_csv("false_negativesNoSMOTE"+outcome+".csv", index=False)

        xgbc.get_feature_importance(experiment_number)


if __name__ == '__main__':
    main()
