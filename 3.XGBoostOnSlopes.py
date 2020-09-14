import os
import json
import pandas as pd
import numpy as np

from Models.XGBoost.XGBoost import run_xgboost_classifier
from Utils.Model import scale, stratified_group_k_fold, generate_trajectory_timeseries, impute
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def main():
    ##1. read configuration file
    configs = json.load(open('Configuration.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']) : os.makedirs(configs['model']['save_dir'])


    grouping = configs['data']['grouping']
    outcome = (configs['data']['classification_outcome'])[0]
    static_features = configs['data']['static_columns']
    dynamic_features = configs['data']['dynamic_columns']

    ##2. read data
    timeseries_path = configs['paths']['clustered_timeseries_path']
    time_series= pd.read_csv(timeseries_path+"SMOTEDTimeSeries/"+outcome+"FlatTimeSeries.csv")

    slopes_df = pd.DataFrame()

    for index, row in time_series.iterrows():
        row_dictionary = {}
        row_dictionary.update({grouping: row[grouping]})
        row_dictionary.update({'cluster_assignment': int(row['cluster_assignment'])})
        row_dictionary.update(zip(static_features, row[static_features]))

        for d in dynamic_features:
            d_columns = [s for s in (time_series.columns).tolist() if d in s]
            row_dictionary.update({d+'_slope': row[d_columns[len(d_columns)-1]]-row[d_columns[0]] })
            row_dictionary.update({d+'_0': row[d_columns[0]] })

        slopes_df = slopes_df.append(row_dictionary, ignore_index=True)

    #slopes_df.columns = row_dictionary.keys()
    slopes_df.to_csv('Run/Data/time_series_slopes_0_mort3D.csv', index=False)

    experiment_number = "1"
    y = time_series[outcome]
    x_columns = ((time_series.columns).tolist())
    x_columns.remove(grouping)
    x_columns.remove(outcome)
    print(x_columns)


    X = time_series[x_columns]
    X.reset_index()
    groups = np.array(time_series[grouping])

    run_xgboost_classifier(X, y, outcome, groups, experiment_number)



if __name__ == '__main__':
    main()
