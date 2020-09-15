import os
import json
import pandas as pd
import numpy as np

from Models.XGBoost.XGBoost import run_xgboost_classifier, run_xgboost_different_datasets
from Utils.Model import scale, stratified_group_k_fold, generate_trajectory_timeseries, impute
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


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
        time_series = pd.read_csv(timeseries_path + "SMOTEDTimeSeries/" + outcome + "FlatTimeSeries.csv")
        non_smoted_time_series = pd.read_csv(timeseries_path + "NonSMOTEDFlatSeries/"+outcome+"FlatTimeSeries.csv")
        experiment_number = "4"
        run_xgboost_different_datasets(time_series, non_smoted_time_series,
                                       outcome, grouping, outcome+experiment_number, experiment_number )


if __name__ == '__main__':
    main()
