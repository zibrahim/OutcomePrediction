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
    clustered_timeseries_path = configs['paths']['clustered_timeseries_path']
    time_series= pd.read_csv(clustered_timeseries_path+"TimeSeriesAggregatedClusteredDeltaTwoDays.csv")


    ##3. impute
    dynamic_features = configs['data']['dynamic_columns']
    grouping = configs['data']['grouping']
    time_series[dynamic_features] = impute(time_series, dynamic_features)

    ##4. generate new features based on delta from baseline
    outcome_columns = configs['data']['classification_outcome']
    static_features = configs['data']['static_columns']

    for outcome in outcome_columns:
        [print('Original DF: Class {} has {} instances after oversampling'.format(label, count))
         for label, count in zip(*np.unique(time_series.loc[:, outcome], return_counts=True))]

        flat_df, timesteps = flatten(time_series, dynamic_features, grouping, static_features, outcome)

        [print('Flat DF: Class {} has {} instances after oversampling'.format(label, count))
         for label, count in zip(*np.unique(flat_df.loc[:,outcome], return_counts=True))]
        timesteps = 18
        smoted_df = smote (flat_df, outcome, grouping)

        [print('Smoted FLAT DF: Class {} has {} instances after oversampling'.format(label, count))
         for label, count in zip(*np.unique(smoted_df.loc[:,outcome], return_counts=True))]

        smoted_timeseries = unflatten(smoted_df, grouping, static_features, outcome, timesteps )

        [print('Unflatted SMOTED DF: Class {} has {} instances after oversampling'.format(label, count))
         for label, count in zip(*np.unique(smoted_timeseries.loc[:,outcome], return_counts=True))]

        smoted_timeseries.to_csv(clustered_timeseries_path+"SMOTEDTimeSeries/"+outcome+"TimeSeries.csv", index=False)
if __name__ == '__main__':
    main()
