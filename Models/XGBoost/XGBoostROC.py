import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,\
    classification_report, auc, roc_curve, brier_score_loss, confusion_matrix

from Utils.Model import stratified_group_k_fold, get_distribution, get_distribution_percentages


def run_xgboost_different_datasets(time_series, non_smoted_time_series,
                                   outcome, grouping, experiment_number):
    y = time_series[outcome]
    y = y.astype(int)

    x_columns = ((time_series.columns).tolist())
    x_columns.remove(grouping)
    x_columns.remove(outcome)

    X = time_series[x_columns]
    X.reset_index()
    groups = np.array(time_series[grouping])


    xgbm=xgb.XGBClassifier(scale_pos_weight=263/73,
                               learning_rate=0.007,
                               n_estimators=100,
                               gamma=0,
                               max_depth=4,
                               min_child_weight=2,
                               subsample=1,
                               eval_metric='error')

    distrs = [get_distribution(y)]
    index = ['Entire set']

    tprs = []
    aucs = []

    cms = np.empty((2, 2))
    mean_fpr = np.linspace(0, 1, 10) #CROSS VALIDATION CHANGE
    plt.figure(figsize=(10, 10))

    stats_df = pd.DataFrame()

    false_X = []
    false_Y = []
    false_IDs = []

    for fold_ind, (training_ind, testing_ind) in enumerate(stratified_group_k_fold(X, y, groups, k=10)) : #CROSS-VALIDATION
            #Train
            training_groups = groups[training_ind]
            training_y = y[training_ind]
            training_X = X.iloc[training_ind]
            xgbm.fit(training_X, training_y)

            #Create Testing Sets
            testing_pool = non_smoted_time_series.loc[~non_smoted_time_series[grouping].isin(training_groups)]

            distrs_percents = [get_distribution_percentages((testing_pool[outcome]).astype(int))]

            length_of_test_set =  len(non_smoted_time_series)/10 #ZI Fix this, number of folds

            number_of_first_class = int(distrs_percents[0][0]* length_of_test_set)
            number_of_second_class = int(distrs_percents[0][1] * length_of_test_set)

            first_half = testing_pool[testing_pool[outcome] == 0]
            second_half = testing_pool[testing_pool[outcome] ==1]
            testing_0 = first_half.sample(n = number_of_first_class, random_state = 1, replace = False)
            testing_1 = second_half.sample(n = number_of_second_class, random_state = 1, replace=False)

            testing =  testing_0.append(testing_1, ignore_index=True)
            testing_ids = testing[grouping]
            testing_X = testing[x_columns]
            testing_X.reset_index()
            testing_y = testing[outcome]
            testing_y = testing_y.astype(int)

            # Train, predict and Plot
            xgbm.fit(testing_X, testing_y)
            y_pred_rt = xgbm.predict_proba(testing_X)[:, 1]

            y_pred_binary = (y_pred_rt > 0.60).astype('int32')

            #Get false negatives:
            for w in range(0,len(testing_y)):
                if (testing_y[w] ==1 and y_pred_binary[w] == 0) or (testing_y[w] ==0 and y_pred_binary[w] ==1):
                    false_Y.append(testing_y[w])
                    false_X.append(testing_X.iloc[w])
                    false_IDs.append(testing_ids[w])

            cm = confusion_matrix(testing_y, y_pred_binary, labels=xgbm.classes_)
            cms += cm
            F1Macro = f1_score(testing_y, y_pred_binary, average='macro')
            F1Micro = f1_score(testing_y, y_pred_binary, average='micro')
            F1Weighted = f1_score(testing_y, y_pred_binary, average='weighted')
            PrecisionMacro = precision_score(testing_y, y_pred_binary, average='macro')
            PrecisionMicro = precision_score(testing_y, y_pred_binary, average='micro')
            PrecisionWeighted = precision_score(testing_y, y_pred_binary, average='weighted')
            RecallMacro = recall_score(testing_y, y_pred_binary, average='macro')
            RecallMicro = recall_score(testing_y, y_pred_binary, average='micro')
            RecallWeighted = recall_score(testing_y, y_pred_binary, average='weighted')
            Accuracy = accuracy_score(testing_y, y_pred_binary)
            ClassificationReport = classification_report(testing_y, y_pred_binary)
            BrierScoreProba = brier_score_loss(testing_y, y_pred_rt)
            BrierScoreBinary = brier_score_loss(testing_y, y_pred_binary)
            fpr, tpr, thresholds = roc_curve(testing_y, y_pred_rt)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            performance_row = {
                "F1-Macro" : F1Macro,
                "F1-Micro" : F1Micro,
                "F1-Weighted" : F1Weighted,
                "Precision-Macro" : PrecisionMacro,
                "Precision-Micro" : PrecisionMicro,
                "Precision-Weighted" : PrecisionWeighted,
                "Recall-Macro" : RecallMacro,
                "Recall-Micro" : RecallMicro,
                "Recall-Weighted" : RecallWeighted,
                "Accuracy" : Accuracy,
                "AUC" : roc_auc,
                "ClassificationReport" : ClassificationReport,
                "BrierScoreProba": BrierScoreProba,
                "BrierScoreBinary": BrierScoreBinary
            }

            stats_df = stats_df.append(performance_row, ignore_index=True)

            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (fold_ind, roc_auc))

            # add to the distribution dataframe, for verification purposes
            distrs.append(get_distribution(training_y))

            index.append(f'training set - fold {fold_ind}')
            distrs.append(get_distribution(testing_y))
            index.append(f'testing set - fold {fold_ind}')

    # Finallise ROC curve
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(outcome+experiment_number, fontsize=18)
    # plt.legend(loc="lower right", prop={'size' : 15})

    stats_path = "Run/Stats/"
    prediction_path = "Run/Prediction/"

    plt.savefig(prediction_path + outcome+experiment_number + "ROC.pdf")

    stats_df.to_csv(stats_path + outcome+experiment_number  + "XGBoost.csv", index=False)

    return false_X, false_Y, false_IDs