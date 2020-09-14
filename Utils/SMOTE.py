import pandas as pd
from kmeans_smote import KMeansSMOTE
import numpy as np

def flatten(time_series, dynamic_features, grouping, static_features, outcome_column):
    #First, create the structure of a flat DF
    newdf = time_series
    newdf.insert(0, grouping+'2', newdf[grouping])
    aggregated_df = newdf.groupby(grouping+'2').aggregate('first')


    timesteps = len(newdf)/len(aggregated_df)
    flat_df = pd.DataFrame()
    for id in aggregated_df[grouping].tolist():
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

        static_feature_values = time_series.loc[time_series[grouping] == id, static_features]
        patient_dict.update(zip(static_features, static_feature_values.iloc[0]))

        outcome_value = time_series.loc[time_series[grouping] == id, outcome_column]

        patient_dict.update({outcome_column: outcome_value.iloc[0]})
        flat_df = flat_df.append(patient_dict, ignore_index=True)

    flat_df.to_csv("flat.csv", index = False)
    return flat_df, timesteps

def smote(target_df, target_outcome, grouping):
    y = target_df[target_outcome]
    target_df.drop(target_outcome, axis=1, inplace=True)

    target_df[grouping] = [float((x.partition('_')[2])) for x in target_df[grouping]]
    #target_df.drop(grouping, axis=1, inplace=True)
    X = target_df

    target_columns = target_df.columns
    #target_columns = target_columns.insert(0, grouping)
    target_columns= target_columns.insert(len(target_columns), target_outcome)
    kmeans_smote = KMeansSMOTE(
        kmeans_args={
            'n_clusters' : 5
        },
        smote_args={
            'k_neighbors' : 10
        }
    )
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, y)
    X_resampled = pd.DataFrame(X_resampled)
    y_resampled = pd.DataFrame(y_resampled)
    frames = [X_resampled, y_resampled]

    total_df = pd.concat(frames, axis=1)
    total_df.columns  = target_columns
    total_df[grouping] = ['p_'+str(x) for x in total_df[grouping]]

    return total_df

def unflatten(flat_df, grouping, static_features, outcome_column, timesteps):
        smoted_timeseries = pd.DataFrame()
        for patient in flat_df[grouping]:
            patient_row = flat_df.loc[flat_df[grouping]==patient]
            patient_row.columns = flat_df.columns
            for step in range(0, timesteps):
                matching_columns = [x for x in patient_row.columns if x.endswith('_'+str(step))]
                dynamic_slot = pd.DataFrame(patient_row[matching_columns])
                dynamic_slot.columns = [x.partition('_')[0] for x in list(dynamic_slot.columns) ]
                dynamic_slot[grouping] = patient
                static_slot = pd.DataFrame(patient_row[static_features])
                static_slot.columns = static_features
                static_slot[grouping] = patient
                static_slot[outcome_column] = patient_row[outcome_column]

                full_slot  = static_slot.merge(dynamic_slot, on=grouping)
                full_slot[grouping] = patient

                slot_dictionary = dict(zip((full_slot.columns).tolist(), (full_slot.values).tolist()[0]))
                smoted_timeseries = smoted_timeseries.append(slot_dictionary, ignore_index=True)

        return smoted_timeseries