import os
import json
import pandas as pd
import numpy as np

from Models.LSTM.Model import LSTMModel
from Utils.Model import scale, stratified_group_k_fold, generate_trajectory_timeseries, impute
from Utils.SMOTE import flatten, smote, unflatten
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def main():

    ##1. read configuration file
    configs = json.load(open('Configuration.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    ##2. read data
    clustered_timeseries_path = configs['paths']['clustered_timeseries_path']
    time_series= pd.read_csv(clustered_timeseries_path+"TimeSeriesAggregatedClusteredDeltaTwoDays.csv")

    #[print('Original DF: Class {} has {} instances after oversampling'.format(label, count))
    # for label, count in zip(*np.unique(time_series.loc[:,'Mortality14Days'], return_counts=True))]
    ##3. impute
    dynamic_features = configs['data']['dynamic_columns']
    grouping = configs['data']['grouping']
    time_series[dynamic_features] = impute(time_series, dynamic_features)

    ##4. generate new features based on delta from baseline
    outcome_columns = configs['data']['classification_outcome']
    baseline_features = configs['data']['baseline_columns']
    static_features = configs['data']['static_columns']

    new_series = generate_trajectory_timeseries(time_series, baseline_features, static_features,
                                                dynamic_features, grouping, outcome_columns)

    new_series.to_csv("newseries.csv", index=False)

    ##5. scale
    normalized_timeseries = scale(new_series, dynamic_features)


    flat_df, timesteps = flatten(new_series, dynamic_features, grouping, static_features, outcome_columns[3])
    #flat_df = pd.read_csv("flat.csv")


    [print('Flat DF: Class {} has {} instances after oversampling'.format(label, count))
     for label, count in zip(*np.unique(flat_df.loc[:,'Mortality14Days'], return_counts=True))]
    timesteps = 18
    smoted_df = smote (flat_df, outcome_columns[3], grouping)

    smoted_df.to_csv("smoted_flat_df.csv", index=False)


    [print('Smoted FLAT DF: Class {} has {} instances after oversampling'.format(label, count))
     for label, count in zip(*np.unique(smoted_df.loc[:,'Mortality14Days'], return_counts=True))]

    smoted_timeseries = unflatten(smoted_df, grouping, static_features, outcome_columns[3], timesteps )


    [print('Unflatted SMOTED DF: Class {} has {} instances after oversampling'.format(label, count))
     for label, count in zip(*np.unique(smoted_timeseries.loc[:,'Mortality14Days'], return_counts=True))]

    smoted_df.to_csv("smoted.csv", index=False)
if __name__ == '__main__':
    main()
