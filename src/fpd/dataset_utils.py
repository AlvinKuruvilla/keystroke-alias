import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def ids():
    integers = list(range(1, 61))
    return [str(i) for i in integers]


def get_df_slice_by_id(data: pd.DataFrame, id: str):
    return data.loc[data["ID"] == id]


def get_target_from_df(data: pd.DataFrame):
    return np.array(list(data.iloc[:, len(list(data.columns[:-1]))]))


def split_into_train_and_test(data: pd.DataFrame, split_percentage: float = 0.7):
    ids_list = ids()
    x_train_holder = []
    y_train_holder = []
    x_test_holder = []
    y_test_holder = []
    for id in ids_list:
        id_df = get_df_slice_by_id(data, id)
        y = get_target_from_df(id_df)
        print(y)
        # input("TARGET")
        assert len(id_df.index) == y.shape[0]
        X_train, X_test, y_train, y_test = train_test_split(
            id_df, y, test_size=split_percentage, random_state=42
        )
        x_train_holder.append(X_train)
        x_test_holder.append(X_test)
        y_train_holder.append(y_train)
        y_test_holder.append(y_test)
    X_train = np.concatenate(x_train_holder, axis=0)
    X_test = np.concatenate(x_test_holder, axis=0)
    y_train = np.concatenate(y_train_holder, axis=0)
    y_test = np.concatenate(y_test_holder, axis=0)
    print("Total DF size:", data.shape)
    print("X_train size:", X_train.shape)
    print("y_train size:", y_train.shape)
    print("X_test size:", X_test.shape)
    print("y_test size:", y_test.shape)
    input("Check length calculations")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )
