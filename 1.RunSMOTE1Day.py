import os
import json
import pandas as pd
import numpy as np

from Utils.Model import scale,impute
from Utils.SMOTE import flatten, smote, unflatten

def main():

    ##1. read configuration file
    configs = json.load(open('Configuration.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    ##2. read data
    timeseries_path = configs['paths']['data_path']
    grouping = configs['data']['grouping']
    outcome_columns = configs['data']['classification_outcome']
    static_features = configs['data']['static_columns']
    dynamic_features = configs['data']['dynamic_columns']
    time_series = pd.read_csv(timeseries_path+"TimeSeriesAggregatedUpto0.csv")
    #3. impute dynamic features
    time_series[dynamic_features] = impute(time_series, dynamic_features)


    #scale
    scaling_columns = static_features+dynamic_features
    normalized_timeseries = scale(time_series, scaling_columns)
    normalized_timeseries.insert(0, grouping, time_series[grouping])
    for col in outcome_columns:
        normalized_timeseries.insert(len(normalized_timeseries.columns), col, time_series[col])

    ##4. generate new features based on delta from baseline
    distribution_df = pd.DataFrame()

    for outcome in outcome_columns:
        distribution_dict = {}

        new_series = normalized_timeseries.copy()
        distribution_dict.update({'outcome':outcome})

        distribution_dict.update({'distribution':[('Class {} -  {} instances '.format(label, count))
         for label, count in zip(*np.unique(time_series.loc[:, outcome], return_counts=True))]})

        distribution_df = distribution_df.append(distribution_dict,ignore_index=True)
        [print('Original DF: Class {} has {} instances after oversampling'.format(label, count))
         for label, count in zip(*np.unique(time_series.loc[:, outcome], return_counts=True))]

        static_flat_columns = static_features
        #static_flat_columns.append('cluster_assignment')
        flat_df, timesteps = flatten(new_series, dynamic_features, grouping,
                                     static_flat_columns, outcome)

        flat_df.to_csv(timeseries_path+"NonSMOTEDTimeSeries/"+outcome+"FlatTimeSeries1Day.csv", index=False)

        [print('Flat DF: Class {} has {} instances after oversampling'.format(label, count))
         for label, count in zip(*np.unique(flat_df.loc[:,outcome], return_counts=True))]
        timesteps = 18
        #smote_columns = static_features+dynamic_features
        #smote_columns.insert(0, grouping)
        #smote_columns.insert(len(smote_columns), outcome)
        #smote_columns = flat_df.columns - baseline_features


        smoted_df = smote (flat_df, outcome, grouping)
        #smoted_df[baseline_features] = flat_df[baseline_features]
        #smoted_df['cluster_assignment'] = flat_df['cluster_assignment']

        [print('Smoted FLAT DF: Class {} has {} instances after oversampling'.format(label, count))
         for label, count in zip(*np.unique(smoted_df.loc[:,outcome], return_counts=True))]

        smoted_df.to_csv(timeseries_path+"SMOTEDTimeSeries/"+outcome+"FlatTimeSeries1Day.csv", index=False)

        smoted_timeseries = unflatten(smoted_df, grouping, static_features, outcome, timesteps )

        [print('Unflatted SMOTED DF: Class {} has {} instances after oversampling'.format(label, count))
         for label, count in zip(*np.unique(smoted_timeseries.loc[:,outcome], return_counts=True))]

        smoted_timeseries.to_csv(timeseries_path+"SMOTEDTimeSeries/"+outcome+"StackedTimeSeries1Day.csv", index=False)
        distribution_df.to_csv("Run/Stats/Distribution"+outcome+".csv")
if __name__ == '__main__':
    main()
