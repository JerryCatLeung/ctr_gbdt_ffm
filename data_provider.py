import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def reformat(df_data):
    arr = df_data.as_matrix()
    X = arr[:, 1:]
    y = arr[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 29)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    return X_train, X_test, y_train, y_test


def get_criteo_data(f_name):
    intnames = []
    catnames = []
    for i in range(13):
        intnames += ['i'+ str(i + 1)]
    for i in range(26):
        catnames += ['c'+ str(i + 1)]
    colnames = ['clicked'] + intnames + catnames

    ds = pd.read_csv(f_name, sep='\t', header=None, names = colnames)
    ds[intnames] = ds[intnames].fillna(ds[intnames].mean())
    ds[intnames] = ds[intnames].astype(int)    

    return reformat(ds)



