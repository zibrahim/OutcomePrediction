import os
import json
import pandas as pd
import numpy as np

from Models.LSTM.Model import LSTMModel
from Utils.Model import scale, stratified_group_k_fold, generate_trajectory_timeseries, impute
from Utils.SMOTE import flatten, smote
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def main():

    ##1. read configuration file
    configs = json.load(open('Configuration.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    ##2. read data
    clustered_timeseries_path = configs['paths']['clustered_timeseries_path']
    time_series= pd.read_csv(clustered_timeseries_path+"TimeSeriesAggregatedClusteredDeltaTwoDays.csv")

    ##3. impute
    dynamic_features = configs['data']['dynamic_columns']
    grouping = configs['data']['grouping']
    time_series[dynamic_features] = impute(time_series, dynamic_features)

    ##4. generate new features based on delta from baseline
    outcome_columns = configs['data']['classification_outcome']
    baseline_features = configs['data']['baseline_columns']
    static_features = configs['data']['static_columns']

    flat_df = flatten(time_series, dynamic_features, grouping, static_features, outcome_columns)

    X, y =  smote (flat_df, outcome_columns[0], outcome_columns, grouping)
    print(X.shape, len(y))
if __name__ == '__main__':
    main()
