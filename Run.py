
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Models.LSTM.Model import Model
from Utils.ModelUtils import scale, stratified_group_k_fold, generate_trajectory_timeseries, impute
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    fig.savefig("foo.pdf", bbox_inches='tight')


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

    new_series = generate_trajectory_timeseries(time_series, baseline_features, static_features,
                                                dynamic_features, grouping, outcome_columns)

    ##5. scale
    normalized_timeseries = scale(new_series, dynamic_features)

    groups = np.array(time_series[grouping])
    X = normalized_timeseries[dynamic_features]
    X.to_csv("X.csv")
    for outcome in configs['data']['classification_outcome']:

        y = time_series[outcome]

        y = y.astype(int)

        model = Model(configs['model']['name'] + outcome)

        model.build_model(configs)

        for ffold_ind, (training_ind, testing_ind) in enumerate(
                stratified_group_k_fold(X, y, groups, k=5)) :  # CROSS-VALIDATION
            training_groups, testing_groups = groups[training_ind], groups[testing_ind]
            this_y_train, this_y_val = y[training_ind], y[testing_ind]

            this_X_train, this_X_val = X.iloc[training_ind], X.iloc[testing_ind]
            y_with_ids = time_series[[grouping, outcome]]

            y_with_ids_training = y_with_ids [y_with_ids[grouping].isin(training_groups)]
            y_with_ids_training = y_with_ids_training.groupby(grouping).first()
            y_with_ids_training = y_with_ids_training[outcome]
            this_y_train = y_with_ids_training.astype(int)

            y_with_ids_testing = y_with_ids [y_with_ids[grouping].isin(testing_groups)]
            y_with_ids_testing = y_with_ids_testing.groupby(grouping).first()
            y_with_ids_testing = y_with_ids_testing[outcome]
            this_y_val = y_with_ids_testing.astype(int)

            assert len(set(training_groups) & set(testing_groups)) == 0

            #(NumberOfExamples, TimeSteps, FeaturesPerStep).

            model.train(
                (this_X_train.values).reshape(-1, 24, 14),
                (this_y_train.values).reshape(-1,1),
                epochs=configs['training']['epochs'],
                batch_size=configs['training']['batch_size'],
                save_dir=configs['model']['save_dir']
            )

            this_X_val.reset_index()

            y_pred_val = model.predict((this_X_val.values).reshape(-1,24,14))
            y_pred_val_binary = (y_pred_val > 0.5).astype('int32')

            print(" ROC AUC: ", roc_auc_score(this_y_val, y_pred_val))
            print(" F1 score Macro: ", f1_score(this_y_val, y_pred_val_binary, average='macro'))
            print(" F1 score Micro: ", f1_score(this_y_val, y_pred_val_binary, average='micro'))
            print(" F1 score Weighted: ", f1_score(this_y_val, y_pred_val_binary, average='weighted'))
            print(" precision score Macro: ", precision_score(this_y_val, y_pred_val_binary, average='macro'))
            print(" precision score Micro: ", precision_score(this_y_val, y_pred_val_binary, average='micro'))
            print(" precision score Weighted: ", precision_score(this_y_val, y_pred_val_binary, average='weighted'))
            print(" recall score Macro: ", recall_score(this_y_val, y_pred_val_binary, average='macro'))
            print(" recall score Micro: ", recall_score(this_y_val, y_pred_val_binary, average='micro'))
            print(" recall score Weighted: ", recall_score(this_y_val, y_pred_val_binary, average='weighted'))

            #plot_results(y_pred_val_binary, this_y_val)
if __name__ == '__main__':
    main()