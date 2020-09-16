import os
import json
import pandas as pd
import numpy as np

from Models.XGBoost.XGBoost import run_xgboost_classifier


def main():
    ##1. read configuration file
    configs = json.load(open('Configuration.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']) : os.makedirs(configs['model']['save_dir'])


    grouping = configs['data']['grouping']
    static_features = configs['data']['static_columns']
    dynamic_features = configs['data']['dynamic_columns']

    outcomes = (configs['data']['classification_outcome'])

    ##2. read data
    timeseries_path = configs['paths']['clustered_timeseries_path']

    for outcome in outcomes:
        time_series= pd.read_csv(timeseries_path+"SMOTEDTimeSeries/"+outcome+"FlatTimeSeries.csv")

        experiment_number = "5"
        y = time_series[outcome]
        x_columns = ((time_series.columns).tolist())
        x_columns.remove(grouping)
        x_columns.remove(outcome)
        print(x_columns)


        X = time_series[x_columns]
        X.reset_index()
        groups = np.array(time_series[grouping])

        run_xgboost_classifier(X, y, outcome+experiment_number, groups, experiment_number)



if __name__ == '__main__':
    main()
