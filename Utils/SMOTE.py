import pandas as pd
from kmeans_smote import KMeansSMOTE
import numpy as np

def flatten(time_series, dynamic_features, grouping, static_features, outcome_columns):
    #First, create the structure of a flat DF
    newdf = time_series
    newdf.insert(0, grouping+'2', newdf[grouping])
    aggregated_df = newdf.groupby(grouping+'2').aggregate('first')

    timesteps = len(newdf)/len(aggregated_df)

    flat_df = pd.DataFrame()
    firstTime = True
    for id in set(aggregated_df[grouping]):
        patient_dict = {}
        patient_chunk = time_series.loc[time_series[grouping] == id, dynamic_features]
        patient_dict.update({grouping: id})

        for timestep in range(0, int(timesteps)) :
            row_dictionary = {}
            for x in patient_chunk.columns :
                datum = patient_chunk.iloc[timestep]
                datum = datum.loc[x]
                if x != grouping:
                    row_dictionary[x+'_'+str(timestep) ] = datum

            patient_dict.update(row_dictionary)
        flat_df = flat_df.append(patient_dict, ignore_index=True)
        if firstTime:
            flat_df.columns = list(patient_dict.keys())
            firstTime = False

    sorted_columns = list(patient_dict.keys())
    sorted_columns.remove((grouping))
    sorted_columns = sorted(sorted_columns, key=lambda x : int(x.split("_")[1]))
    sorted_columns.insert(0, grouping)

    flat_df = flat_df.reindex(columns=sorted_columns)

    column_list = static_features
    column_list.insert(0,grouping)
    column_list.extend(outcome_columns)

    flat_df = pd.merge(flat_df, aggregated_df.loc[:,column_list], on= grouping, how='inner')

    flat_df.to_csv("flat.csv", index = False)
    return flat_df

def smote(target_df, target_outcome, outcome_columns, grouping):

    print("initial target df shape", target_df.shape)
    y = target_df[target_outcome]
    target_df.drop(outcome_columns, axis=1, inplace=True)
    ids = target_df[grouping]
    target_df.drop(grouping, axis=1, inplace=True)
    X = target_df

    kmeans_smote = KMeansSMOTE(
        kmeans_args={
            'n_clusters' : 100
        },
        smote_args={
            'k_neighbors' : 10
        }
    )
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, y)
    X_resampled = pd.DataFrame(X_resampled)
    print(" printing y's length before dfing it", len(y_resampled))
    y_resampled = pd.DataFrame(y_resampled)
    ids = pd.DataFrame(ids)
    frames = [ids, X_resampled, y_resampled]
    total_df = pd.concat(frames, axis=1)

    [print('Class {} has {} instances after oversampling'.format(label, count))
     for label, count in zip(*np.unique(y_resampled, return_counts=True))]

    print(type(X_resampled))
    print(" shapes: x, ", X_resampled.shape, "y: ", len(y_resampled), "total df: ", total_df.shape)
    return X_resampled, y_resampled

