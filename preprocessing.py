import numpy as np
from collections import Counter

def get_nan(row):
    Nan_index = []
    for i in range(len(row)):
        if np.isnan(row[i]):
            Nan_index.append(i)
    return Nan_index

def get_mean(row,num_nan):
    Sum = 0
    for e in row:
        if not np.isnan(e):
            Sum += e
    mean = Sum/(len(row) - num_nan)
    return mean

def get_mode(row,num_nan):
    filtered_row = []
    for e in row:
        if not np.isnan(e):
            filtered_row.append(e)
    mode = Counter(filtered_row).most_common(1)
    return mode[0][0]

def replace_null_values_with_mean(X,col):
    X = X.T
    nan = get_nan(X[col])
    if nan:
        mean = round(get_mean(X[col],len(nan)),5)
        for j in nan:
            X[col][j] = mean
    return X.T

def replace_null_values_with_mode(X,col):
    X = X.T
    nan = get_nan(X[col])
    if nan:
        mode = round(get_mode(X[col],len(nan)),5)
        for j in nan:
            X[col][j] = mode
    return X.T


def min_max_normalize(X, column_indices):
    X = X.T
    mmn_X = []
    for i in range(len(X)):
        if i in column_indices:
            row = X[i]
            row = (row - np.min(row))/(np.max(row) - np.min(row))
            mmn_X.append(row)
        else:
            mmn_X.append(X[i])
    mmn_X = np.array(mmn_X)

    return mmn_X.T

def apply_one_hot_encoding(X):
    label_dict = dict()
    label_val = 0
    sorted_X = np.sort(X)
    for l in sorted_X:
        if l not in label_dict.keys():
            label_dict[l] = label_val
            label_val += 1
    encoded_X = np.zeros((len(X),len(label_dict)))
    for i in range(len(X)):
        encoded_X[i][label_dict[X[i]]] = 1
    return encoded_X.astype(int)


def convert_given_features_to_one_hot(X, column_indices):
    X = X.T
    converted_X = []
    for i in range(len(X)):
        if i in column_indices:
            temp = apply_one_hot_encoding(X[i])
            temp = list(temp.T)
            for r in temp:
                converted_X.append(r)
        else:
            converted_X.append(X[i])
        
    converted_X = np.array(converted_X)
    return converted_X.T


def preprocess(X):
    feature = { "sex":0 , "fever_frequency":1 , "blood_sugar":2 , "travelled_place":3 , "breathing_difficulty_level":4 , "age":5 , "immunity_level":6 }
    #print("initially",X.shape)
    X = replace_null_values_with_mean(X,feature["age"])
    X = replace_null_values_with_mode(X,feature["travelled_place"])
    #print("handled null values",X.shape)
    min_max_features = [feature["blood_sugar"],feature["age"]]
    X = min_max_normalize(X,min_max_features)
    #print("after min_max",X.shape)
    one_hot_features = [feature["fever_frequency"],feature["travelled_place"],feature["breathing_difficulty_level"],feature["immunity_level"]]
    X = convert_given_features_to_one_hot(X,one_hot_features)
    #print("after one hot encoding",X.shape)
    return X
