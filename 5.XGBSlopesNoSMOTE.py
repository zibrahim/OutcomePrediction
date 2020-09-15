import os
import json
import pandas as pd
import numpy as np

from Models.XGBoost.XGBoost import run_xgboost_classifier
from Utils.Model import scale, stratified_group_k_fold, generate_trajectory_timeseries, impute
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from Utils.SMOTE import flatten


def main():
    ##1. read configuration file
    configs = json.load(open('Configuration.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    ##2. read data
    timeseries_path = configs['paths']['clustered_timeseries_path']
    grouping = configs['data']['grouping']
    outcome_columns = configs['data']['classification_outcome']
    static_features = configs['data']['static_columns']
    dynamic_features = configs['data']['dynamic_columns']
    baseline_features = configs['data']['baseline_columns']
    time_series= pd.read_csv(timeseries_path+"TimeSeriesAggregatedClusteredDeltaTwoDays.csv")
    time_series.drop(baseline_features, axis='columns', inplace=True)

    #3. impute dynamic features
    time_series[dynamic_features] = impute(time_series, dynamic_features)


    #scale
    scaling_columns = static_features+dynamic_features
    normalized_timeseries = scale(time_series, scaling_columns)
    normalized_timeseries.insert(0, grouping, time_series[grouping])
    normalized_timeseries.insert(len(normalized_timeseries.columns),'cluster_assignment', time_series['cluster_assignment'])
    for col in outcome_columns:
        normalized_timeseries.insert(len(normalized_timeseries.columns), col, time_series[col])

    ## flatten

    static_flat_columns = static_features
    static_flat_columns.append('cluster_assignment')

    for outcome in outcome_columns :
        tobe_flattened = normalized_timeseries.copy()
        flat_df, timesteps = flatten(tobe_flattened, dynamic_features, grouping,
                                     static_flat_columns, outcome)
        slopes_df = pd.DataFrame()

        for index, row in flat_df.iterrows() :
            row_dictionary = {}
            row_dictionary.update({grouping : row[grouping]})
            row_dictionary.update({'cluster_assignment' : int(row['cluster_assignment'])})
            row_dictionary.update(zip(static_features, row[static_features]))

            for d in dynamic_features :
                d_columns = [s for s in (flat_df.columns).tolist() if d in s]
                row_dictionary.update({d + '_slope' : row[d_columns[len(d_columns) - 1]] - row[d_columns[0]]})
                row_dictionary.update({d + '_0' : row[d_columns[0]]})

            slopes_df = slopes_df.append(row_dictionary, ignore_index=True)

        experiment_number = "3"
        y = flat_df[outcome]
        x_columns = ((flat_df.columns).tolist())
        x_columns.remove(grouping)
        x_columns.remove(outcome)
        print(x_columns)

        X = flat_df[x_columns]
        X.reset_index()
        groups = np.array(flat_df[grouping])

        run_xgboost_classifier(X, y, outcome + experiment_number, groups, experiment_number)


if __name__ == '__main__' :
    main()
