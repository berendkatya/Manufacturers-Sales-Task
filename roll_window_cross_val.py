import numpy as np
import pandas as pd

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def smape(true, predict):
    return (100/len(true)) * np.sum(2 * np.abs(predict - true) / (np.abs(true) + np.abs(predict)))

def rolling_window_score(model, df, target, list_of_features, date, train_size, test_size, step):
    block_size = train_size + test_size
    timestamps = [list(df[date])[i: i + block_size] for i in range(0, df[date].shape[0] - block_size + 1, step)]
    scores = []
    for i in range(len(timestamps)):
        df_train = df[df[date].isin(timestamps[i][:-test_size])]
        df_test = df[df[date].isin(timestamps[i][-test_size:])]
        model.fit(df_train[list_of_features], df_train[target])
        result = model.predict(df_test[list_of_features])
        scores.append(smape(df_test[target], result))
    return scores

def rolling_window_predict(model, df, target, list_of_features, date, train_size, test_size):
    step = test_size
    block_size = train_size + test_size
    timestamps = [list(df[date])[i: i + block_size] for i in range(0, df[date].shape[0] - block_size + 1, step)]
    df['predict'] = np.NaN
    for i in range(len(timestamps)):
        df_train = df[df[date].isin(timestamps[i][:-test_size])]
        df_test = df[df[date].isin(timestamps[i][-test_size:])]
        model.fit(df_train[list_of_features], df_train[target])
        df['predict'][df[date].isin(timestamps[i][-test_size:])] = model.predict(df_test[list_of_features]) 
    return df
        