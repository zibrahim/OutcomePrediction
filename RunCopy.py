
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Models.LSTM.Model import Model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from Utils.Model import scale, stratified_group_k_fold
from sklearn.metrics import roc_auc_score

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    fig.savefig("foo.pdf", bbox_inches='tight')


def main():
    configs = json.load(open('Configuration.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    clustered_timeseries_path = configs['paths']['clustered_timeseries_path']
    time_series= pd.read_csv(clustered_timeseries_path+"TimeSeriesAggregatedClusteredDeltaTwoDays.csv")
    #print(time_series.shape)
       # configs['data']['train_test_split'],  #the split
        #configs['data']['columns_dynamic'] # the columns

    #Impute and Scale Data


    dynamic_features = configs['data']['dynamic_columns']
    grouping = configs['data']['grouping']
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(time_series[dynamic_features])
    time_series[dynamic_features] = imp.transform(time_series[dynamic_features])
    time_series = scale(time_series, dynamic_features)

    X = time_series[dynamic_features]
    groups = np.array(time_series[grouping])

    for outcome in configs['data']['classification_outcome']:
        y = time_series[outcome]

        y = y.astype(int)

        model = Model(configs['model']['name'] + outcome)

        model.build_model(configs)

        for ffold_ind, (training_ind, testing_ind) in enumerate(
                stratified_group_k_fold(X, y, groups, k=10)) :  # CROSS-VALIDATION
            training_groups, testing_groups = groups[training_ind], groups[testing_ind]
            this_y_train, this_y_val = y[training_ind], y[testing_ind]
            #ungrouped_y_train = time_series[[grouping, outcome]]
            #ungrouped_y_train = ungrouped_y_train[training_ind]
            #grouped_y_train = ungrouped_y_train.groupby(grouping).first()

            #ungrouped_y_val = [grouping, this_y_val]
            this_X_train, this_X_val = X.iloc[training_ind], X.iloc[testing_ind]
            y_with_ids = time_series[[grouping, outcome]]
            y_with_ids = y_with_ids [y_with_ids[grouping].isin(testing_groups)]
            y_with_ids = y_with_ids.groupby(grouping).first()
            y_true = y_with_ids[outcome]
            y_true = y_true.astype(int)

            assert len(set(training_groups) & set(testing_groups)) == 0

            #(NumberOfExamples, TimeSteps, FeaturesPerStep).
            model.train(
                (this_X_train.values).reshape(-1, 24, 35),
                (this_y_train.values).reshape(-1,24),
                epochs=configs['training']['epochs'],
                batch_size=configs['training']['batch_size'],
                save_dir=configs['model']['save_dir']
            )

            this_X_val.reset_index()
            #predictions = model.predict_sequences_multiple(this_X_val, configs['data']['sequence_length'],
                                                           #24)
            y_pred_val = model.predict((this_X_val.values).reshape(-1,24,35))
            y_pred_val_binary = (y_pred_val > 0.5).astype('int32')

            x = y_pred_val.reshape(-1,24,1)
            print(" NEW X SHAPE: ", x.shape)
            #print(" predictions")
            print("Predicted: ", np.unique(y_pred_val_binary), len(y_pred_val_binary), y_pred_val_binary.shape)
            #print(" All validation ", np.unique(this_y_val), len(this_y_val), this_y_val.shape)
            #print("Validation for the subset: ", np.unique(y_true), len(y_true), y_true.shape)


            print(" DOING ROC FOR Y TRU AND Y PREDICTE")
            print(" Y True: ", y_true.shape)
            print(" Y predicted: ", y_pred_val_binary.shape)
            print(" ROC AUC: ", roc_auc_score(y_true, y_pred_val_binary[:,20]))

            #plot_results(y_pred_val_binary, this_y_val)
if __name__ == '__main__':
    main()