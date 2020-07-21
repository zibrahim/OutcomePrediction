from collections import Counter, defaultdict
import random
import numpy as np
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from keras.activations import *

lstm_dict={}
lstm_dict['activation'] = ['sigmoid']
lstm_dict['batch'] = ['all']
lstm_dict['data'] = ['balanced', 'unchanged']
lstm_dict['dropout'] = [0.0, 0.2, 0.4, 0.6]
lstm_dict['layers'] = [1, 2]
lstm_dict['masking'] = [True]
lstm_dict['optimizer'] = ['RMSprop', 'Adagrad']
lstm_dict['outcome'] = ['ITUAdmission7Days', 'ITUAdmission14Days', 'ITUAdmission30Days',
                         'Mortality7Days','Mortality14Days','Mortality30Days']
lstm_dict['units'] = [1, 4, 8, 16, 32, 64, 128]

lstm_running_params = {}
lstm_running_params['monitor_checkpoint'] = ['val_matthews', 'max']
lstm_running_params['monitor_early_stopping'] = ['val_matthews', 'max']
lstm_running_params['early_stopping'] = True
lstm_running_params['save_checkpoint'] = True
lstm_running_params['seed'] = 42
lstm_running_params['n_splits'] = 10
lstm_running_params['n_epochs'] = 1000
lstm_running_params['patience'] = 200

def stratified_group_k_fold ( X, y, groups, k, seed=None ) :
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda : np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups) :
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda : np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold ( y_counts, fold ) :
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num) :
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x : -np.std(x[1])) :
        best_fold = None
        min_eval = None
        for i in range(k) :
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval :
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k) :
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def get_distribution ( y_vals ) :
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())
    return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]

def lstm_precision(y_true, y_pred):
 true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
 predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
 precision = true_positives / (predicted_positives + K.epsilon())
 return precision

def lstm_recall(y_true, y_pred):
 true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
 possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
 recall = true_positives / (possible_positives + K.epsilon())
 return recall

def lstm_matthews(y_true, y_pred):
 y_pred_pos = K.round(K.clip(y_pred, 0, 1))
 y_pred_neg = 1 - y_pred_pos
 y_pos = K.round(K.clip(y_true, 0, 1))
 y_neg = 1 - y_pos
 tp = K.sum(y_pos * y_pred_pos)
 tn = K.sum(y_neg * y_pred_neg)
 fp = K.sum(y_neg * y_pred_pos)
 fn = K.sum(y_pos * y_pred_neg)
 numerator = (tp * tn - fp * fn)
 denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 return numerator / (denominator + K.epsilon())

def generate_balanced_arrays(X_train, y_train):
 while True:
  positive = np.where(y_train==1)[0].tolist()
  print("POSITIVE: ")
  print(positive)
  negative = np.random.choice(np.where(y_train==0)[0].tolist(),size = len(positive), replace = False)
  print("NEGATIVE")
  print(negative)

  balance = np.concatenate((positive, negative), axis=0)
  print(" BALANCE: ")

  print("BALANCE CLASS: ", type(balance))
  print(balance)
  np.random.shuffle(balance)
  input = X_train.iloc[balance, :]
  #input = X_train[balance,:]

  print("INPUTTT")
  print(input)

  print(" CLASS OF Y: ", type(y_train))
  target = y_train.iloc[balance]
  print("SHAPE OF TRAIN: ", input.shape, " SHAPE OF TARGET: ", target.shape, len(target))
  print(" COLUMN NAMES OF TRAIN: ", input.columns)
  yield input, target

def scale(df, scale_columns):
    for col in scale_columns:
        series = df[col]
        values = series.values
        values = values.reshape((len(values), 1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(values)
        normalised = scaler.transform(values)
        df['col'] = normalised

    return df
