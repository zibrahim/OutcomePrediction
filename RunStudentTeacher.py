
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Models.LSTM.Model import LSTMModel
from Models.XGBoost.Model import XGBoostModel
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
    baseline_features = configs['data']['baseline_columns']
    static_features = configs['data']['static_columns']

    new_series = generate_trajectory_timeseries(time_series, baseline_features, static_features,
                                                dynamic_features, grouping, outcome_columns)

    ##5. scale
    normalized_timeseries = scale(new_series, dynamic_features)

    groups = np.array(time_series[grouping])
    X = normalized_timeseries[dynamic_features]
    X_student = new_series
    X_student = X_student[static_features]
    X_student[grouping] = new_series[grouping]
    X_student[grouping+"2"] = X_student[grouping]
    X_student = X_student.groupby(grouping+"2").first()
    X_student['TeacherOutcome'] = 0

    print(" AFTER AGGREGATION, DIM OF X_STUDENT: ", X_student.shape)
    ##6. Training/Prediction for all outcomes.
    for outcome in configs['data']['classification_outcome']:

        outcome_df = pd.DataFrame()

        number_of_features = configs['data']['sequence_length']
        batch_size = configs['training']['batch_size']

        y = time_series[outcome]
        y = y.astype(int)
        teacher_model = LSTMModel(configs['model']['name'] + outcome)
        teacher_model.build_model(configs)

        student_model = XGBoostModel(configs['model']['name'] + outcome)

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
            print("COLUMN NAMES OF Y WITH IDS TESTING: ")
            print(y_with_ids_testing.columns)
            print(type(y_with_ids_testing))
            y_with_ids_testing[grouping+"2"] = y_with_ids_testing[grouping]
            print(type(y_with_ids_testing))

            y_with_ids_testing = y_with_ids_testing.groupby(grouping+"2").first()
            print(type(y_with_ids_testing))
            print(y_with_ids_testing.columns)

            this_y_ids = y_with_ids_testing[grouping]
            this_y_val = y_with_ids_testing[outcome]
            this_y_val = this_y_val.astype(int)

            assert len(set(training_groups) & set(testing_groups)) == 0

            #(NumberOfExamples, TimeSteps, FeaturesPerStep).

            teacher_model.train(
                (this_X_train.values).reshape(-1, batch_size, number_of_features),
                (this_y_train.values).reshape(-1,1),
                epochs=configs['training']['epochs'],
                batch_size=batch_size,
                save_dir=configs['model']['save_dir']
            )

            this_X_val.reset_index()

            y_pred_val_teacher = teacher_model.predict((this_X_val.values).reshape(-1,batch_size,number_of_features))


            print("PRINTING SOME USEFUL INFO")

            print(" STUDENT IDS: ")
            print(X_student[grouping])
            print(" IDS IN THIS RUN")
            print(this_y_ids)
            X_student.loc[X_student[grouping].isin(this_y_ids),'TeacherOutcome'] = y_pred_val_teacher

            y_pred_val_binary = (y_pred_val_teacher > 0.5).astype('int32')

            print(" ROC AUC: ", roc_auc_score(this_y_val, y_pred_val_teacher))


            F1Micro = f1_score(this_y_val, y_pred_val_binary, average='micro')
            F1Weighted = f1_score(this_y_val, y_pred_val_binary, average='weighted')
            PrecisionMicro =  precision_score(this_y_val, y_pred_val_binary, average='micro')
            PrecisionWeighted =  precision_score(this_y_val, y_pred_val_binary, average='weighted')
            RecallMicro =  recall_score(this_y_val, y_pred_val_binary, average='micro')
            RecallWeighted =  recall_score(this_y_val, y_pred_val_binary, average='weighted')

            print(F1Micro, F1Weighted, PrecisionMicro, PrecisionWeighted, RecallMicro, RecallWeighted)


            performance_row = {
                "F1-Micro" :F1Micro,
                "F1-Weighted" : F1Weighted,
                "Precision-Micro" : PrecisionMicro,
                "Precision-Weighted": PrecisionWeighted,
                "Recall-Micro": RecallMicro,
                "Recall-Weighted": RecallWeighted
            }

            outcome_df = outcome_df.append(performance_row, ignore_index=True)
        outcome_df.to_csv(outcome+"Teacher.csv")
        X_student.to_csv(outcome+"StudentTrainingData.csv")
        print(" STUDENT TRAINING COLUMN NAMES : ")
        print(X_student.columns)
        X_student = X_student[['Age','SxToAdmit','NumComorbidities',
                                      'cluster_assignment', 'TeacherOutcome']]

        print("CLASSESSS OF TRAINING INPUT:")
        print(" Xstudnt", X_student.shape, type(X_student))
        print(" y val", len(this_y_val), type(this_y_val))
        print(" outcome: ", outcome)
        print(" configs", type(configs))
        ##ZI MAKE SURE YS CORRESPOND TO THE XS. DON'T JUST USE Y IN THIS CALL
        student_model.train(X_student, y, outcome, configs)

    #plot_results(y_pred_val_binary, this_y_val)
if __name__ == '__main__':
    main()