import pandas as pd
import os

def split_dataset(data: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15):
    """
    Split a dataset into train, validation and test sets
    :param data: pandas DataFrame
    :param train_size: float
    :param val_size: float
    :param test_size: float
    :return: pandas DataFrame, pandas DataFrame, pandas DataFrame
    """
    assert train_size + val_size + test_size == 1
    train = data.iloc[:int(len(data) * train_size)]
    val = data.iloc[int(len(data) * train_size):int(len(data) * (train_size + val_size))]
    test = data.iloc[int(len(data) * (train_size + val_size)):]
    return train, val, test

dir = 'eurusd_15'
files = os.listdir(dir)

for file in files:
    if file.endswith(".csv"):
        data = pd.read_csv(f"eurusd_15\\{file}")
        train, val, test = split_dataset(data)
        if os.path.exists('data\\eurusd_15_train') == False:
            os.makedirs('data\\eurusd_15_train')
        if os.path.exists('data\\eurusd_15_val') == False:
            os.makedirs('data\\eurusd_15_val')
        if os.path.exists('data\\eurusd_15_test') == False:
            os.makedirs('data\\eurusd_15_test')
        train.to_csv(f"data\\eurusd_15_train\\{file}", index=False)
        val.to_csv(f"data\\eurusd_15_val\\{file}", index=False)
        test.to_csv(f"data\\eurusd_15_test\\{file}", index=False)
        print(f"Split {file} into train, val and test sets")