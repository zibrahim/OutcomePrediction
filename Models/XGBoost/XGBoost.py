import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, \
    classification_report, auc, brier_score_loss, precision_recall_curve
from Utils.Model import stratified_group_k_fold, get_distribution, get_distribution_percentages, \
    get_distribution_scalars


class XGBoostClassifier():

    def get_features( self ):
        x_columns = ((self.time_series.columns).tolist())
        x_columns.remove(self.grouping)
        x_columns.remove(self.outcome)
        return x_columns

    def get_feature_importance ( self, experiment_number) :
        prediction_path = "Run/Prediction/"

        f ='gain'
        important_features = self.model.get_booster().get_score(importance_type=f)

        print(important_features)
        ax = xgb.plot_importance(self.model)
        ax.figure.tight_layout()

        ax.figure.savefig(prediction_path+self.outcome+experiment_number+".png")


    def __init__(self, time_series, non_smoted_time_series,outcome, grouping):

        self.time_series = time_series
        self.non_smoted_time_series = non_smoted_time_series
        self.outcome = outcome
        self.grouping = grouping
        class_distributions = [get_distribution_scalars((non_smoted_time_series[outcome]).astype(int))]

        class_weights = class_distributions[0][0] / class_distributions[0][1]

        self.model = xgb.XGBClassifier(scale_pos_weight=class_weights,
                                 learning_rate=0.007,
                                 n_estimators=100,
                                 gamma=0,
                                 max_depth=4,
                                 min_child_weight=2,
                                 subsample=1,
                                 eval_metric='error')


    def run_xgb(self, experiment_number, smote):
        y = self.time_series[self.outcome]
        y = y.astype(int)

        x_columns = ((self.time_series.columns).tolist())
        x_columns.remove(self.grouping)
        x_columns.remove(self.outcome)

        X = self.time_series[x_columns]
        X.reset_index()
        groups = np.array(self.time_series[self.grouping])





        distrs = [get_distribution(y)]
        index = ['Entire set']

        prs = []
        aucs = []
        mean_recall = np.linspace(0, 1, 100)
        threshold_indices = []
        i = 0

        plt.figure(figsize=(10, 10))

        stats_df = pd.DataFrame()

        false_X = []
        false_Y = []
        false_IDs = []

        for fold_ind, (training_ind, testing_ind) in enumerate(stratified_group_k_fold(X, y, groups, k=10)) : #CROSS-VALIDATION
            #Train

                training_groups = groups[training_ind], groups[testing_ind]
                training_y, testing_y = y[training_ind], y[testing_ind]
                training_X, testing_X = X.iloc[training_ind], X.iloc[testing_ind]
                self.model.fit(training_X, training_y)

            #default, before looking into whether the dataset has been smoted
                testing_ids = groups[testing_ind]
            #Create Testing Sets
                if smote == True:
                    testing_pool = self.non_smoted_time_series.loc[~self.non_smoted_time_series[self.grouping].isin(training_groups)]
                    distrs_percents = [get_distribution_percentages((testing_pool[self.outcome]).astype(int))]

                    length_of_test_set =  len(self.non_smoted_time_series)/10 #ZI Fix this, number of folds
                #length_of_test_set =  len(training_X)/9 #ZI Fix this, number of folds

                    number_of_first_class = int(distrs_percents[0][0]* length_of_test_set)
                    number_of_second_class = int(distrs_percents[0][1] * length_of_test_set)

                    first_half = testing_pool[testing_pool[self.outcome] == 0]
                    second_half = testing_pool[testing_pool[self.outcome] ==1]
                    testing_0 = first_half.sample(n = number_of_first_class, random_state = 1, replace = False)
                    testing_1 = second_half.sample(n = number_of_second_class, random_state = 1, replace=False)

                    testing =  testing_0.append(testing_1, ignore_index=True)
                    testing_ids = testing[self.grouping]
                    testing_X = testing[x_columns]
                    testing_X.reset_index()
                    testing_y = testing[self.outcome]
                    testing_y = testing_y.astype(int)

                # Train, predict and Plot
                self.model.fit(testing_X, testing_y)
                y_pred_rt = self.model.predict_proba(testing_X)[:, 1]

                precision, recall, thresholds = precision_recall_curve(testing_y, y_pred_rt)
                fscore = (2 * precision * recall) / (precision + recall)
                ix = np.argmax(fscore)
                threshold_indices.append(ix)

                y_pred_binary = (y_pred_rt > thresholds[ix]).astype('int32')

            #Get false negatives:

                if smote==False:
                    for w in range(0,len(testing_y)):
                        if (testing_y.iloc[w] ==1 and y_pred_binary[w] == 0) or (testing_y.iloc[w] ==0 and y_pred_binary[w] ==1):
                            false_Y.append(testing_y.iloc[w])
                            false_X.append(testing_X.iloc[w])
                            false_IDs.append(testing_ids[w])

                else:
                    for w in range(0,len(testing_y)):
                        if (testing_y[w] ==1 and y_pred_binary[w] == 0) or (testing_y[w] ==0 and y_pred_binary[w] ==1):
                            false_Y.append(testing_y[w])
                            false_X.append(testing_X.iloc[w])
                            false_IDs.append(testing_ids[w])

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


                prs.append(np.interp(mean_recall, precision, recall))
                pr_auc = auc(recall, precision)
                aucs.append(pr_auc)
                i += 1

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
                    "ClassificationReport" : ClassificationReport,
                    "BrierScoreProba": BrierScoreProba,
                    "BrierScoreBinary": BrierScoreBinary
                }

                stats_df = stats_df.append(performance_row, ignore_index=True)

            # add to the distribution dataframe, for verification purposes
                distrs.append(get_distribution(training_y))

                index.append(f'training set - fold {fold_ind}')
                distrs.append(get_distribution(testing_y))
                index.append(f'testing set - fold {fold_ind}')



        plt.plot([0, 1], [1, 0], linestyle='--', lw=3, label='Luck', alpha=.8)

        mean_precision = np.mean(prs, axis=0)
        mean_auc = auc(mean_recall, mean_precision)
        plt.plot(mean_precision, mean_recall, color='navy',
             label=r' Mean AUCPR = %0.3f' % mean_auc,
             lw=4)

        plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)

        stats_path = "Run/Stats/"
        prediction_path = "Run/Prediction/"

        plt.legend(prop={'size' : 10}, loc=4)
        plt.savefig(prediction_path+"ROC"+self.outcome+experiment_number+".pdf", bbox_inches='tight')

        stats_df.to_csv(stats_path + self.outcome+experiment_number  + "XGB.csv", index=False)

        #f = plt.figure()

        #rf_shap_values = shap.KernelExplainer(self.model.predict, testing_X)
        #shap.summary_plot(rf_shap_values, testing_X)

        #f.savefig(prediction_path+"SHAP"+self.outcome+experiment_number+".pdf", bbox_inches='tight')

        return false_X, false_Y, false_IDs

