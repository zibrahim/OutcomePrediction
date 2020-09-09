import os
import json
import pandas as pd
import numpy as np

from Models.LSTM.Model import LSTMModel
from Utils.Model import scale, stratified_group_k_fold, generate_trajectory_timeseries, impute
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
    print(" OUTCOME COLUMNS: ", outcome_columns)
    baseline_features = configs['data']['baseline_columns']
    static_features = configs['data']['static_columns']

    new_series = generate_trajectory_timeseries(time_series, baseline_features, static_features,
                                                dynamic_features, grouping, outcome_columns)

    ##5. scale
    normalized_timeseries = scale(new_series, dynamic_features)

    groups = np.array(time_series[grouping])
    X = normalized_timeseries[dynamic_features]

    ##6. Training/Prediction for all outcomes.
    for outcome in configs['data']['classification_outcome']:

        outcome_df = pd.DataFrame()

        number_of_features = configs['data']['sequence_length']
        batch_size = configs['training']['batch_size']

        y = time_series[outcome]
        y = y.astype(int)
        model = LSTMModel(configs['model']['name'] + outcome)
        model.build_model(configs)

        for ffold_ind, (training_ind, testing_ind) in enumerate(
                stratified_group_k_fold(X, y, groups, k=5)) :  # CROSS-VALIDATION
            training_groups, testing_groups = groups[training_ind], groups[testing_ind]
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

            print(" training data length: x: ", this_X_train.shape, " Y: ", this_y_train.shape)
            model.train(
                (this_X_train.values).reshape(-1, batch_size, number_of_features),
                (this_y_train.values).reshape(-1,1),
                epochs=configs['training']['epochs'],
                batch_size=batch_size,
                save_dir=configs['model']['save_dir']
            )

            this_X_val.reset_index()

            y_pred_val = model.predict((this_X_val.values).reshape(-1,batch_size,number_of_features))
            y_pred_val_binary = (y_pred_val > 0.5).astype('int32')

            print(" ROC AUC: ", roc_auc_score(this_y_val, y_pred_val))

            F1Macro = f1_score(this_y_val, y_pred_val_binary, average='macro')
            F1Micro = f1_score(this_y_val, y_pred_val_binary, average='micro')
            F1Weighted = f1_score(this_y_val, y_pred_val_binary, average='weighted')
            PrecisionMacro =  precision_score(this_y_val, y_pred_val_binary, average='macro')
            PrecisionMicro =  precision_score(this_y_val, y_pred_val_binary, average='micro')
            PrecisionWeighted =  precision_score(this_y_val, y_pred_val_binary, average='weighted')
            RecallMacro =  recall_score(this_y_val, y_pred_val_binary, average='macro')
            RecallMicro =  recall_score(this_y_val, y_pred_val_binary, average='micro')
            RecallWeighted =  recall_score(this_y_val, y_pred_val_binary, average='weighted')

            performance_row = {
                "F1-Macro" : F1Macro,
                "F1-Micro" :F1Micro,
                "F1-Weighted" : F1Weighted,
                "Precision-Macro" : PrecisionMacro,
                "Precision-Micro" : PrecisionMicro,
                "Precision-Weighted": PrecisionWeighted,
                "Recall-Macro" : RecallMacro,
                "Recall-Micro": RecallMicro,
                "Recall-Weighted": RecallWeighted
            }

            outcome_df = outcome_df.append(performance_row, ignore_index=True)
        outcome_df.to_csv("Outcomes/"+outcome+".csv")

if __name__ == '__main__':
    main()
