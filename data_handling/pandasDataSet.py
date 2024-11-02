import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import List
from sklearn.preprocessing import MinMaxScaler


def list_files(directory: str) -> List[str]:
    return [os.path.join(directory, file_) for file_ in os.listdir(directory) if os.path.isfile(os.path.join(directory, file_))]


def check_tensor_shapes(tensors):
    shapes = [tensor.shape for tensor in tensors]
    return all(x == shapes[0] for x in shapes)


class PandasDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, window_size: int, cols=['Close'], target_cols=['Close'], normalize=False, differentiate=False, prediction_size=1):
        self.dataframe = dataframe
        self.window_size = window_size
        self.cols = cols
        self.target_cols = target_cols
        self.should_normalize = normalize
        self.should_differentiate = differentiate
        self.prediction_size = prediction_size
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        if self.should_normalize:
            self.normalize()
        if self.should_differentiate:
            self.differentiate()

    def __getitem__(self, index: int) -> tuple:
        # select cols for x, and target_cols for y
        x = self.dataframe.iloc[index:index+self.window_size][self.cols].values
        y = self.dataframe.iloc[index+self.window_size:index +
                    self.window_size+self.prediction_size][self.target_cols].values        
        return np.array(x), np.array(y)

    def __len__(self) -> int:
        return len(self.dataframe) - self.window_size - self.prediction_size

    def collate_fn(self, batch) -> List[torch.Tensor]:
        return [torch.tensor(np.array(x)).float() for x in zip(*batch)]

    def normalize(self):
        # self.dataframe = (self.dataframe - self.dataframe.mean()) / self.dataframe.std()
        self.dataframe[self.cols] = self.scaler.fit_transform(self.dataframe[self.cols])

    def denormalize(self, data):
        return self.scaler.inverse_transform(data)

    def differentiate(self):
        self.dataframe[self.cols] = self.dataframe[self.cols].diff(periods=1).dropna()

    @property
    def length(self):
        return len(self)

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        self._scaler = value


class DistributedDataset(Dataset):
    def __init__(self, directory: str, window_size: int, normalize: bool = False, differentiate: bool = False, cols=['Close'], target_cols=['Close'], prediction_size=1):
        self.datasets = []
        self.idx_dist = []
        self.files = list_files(directory)
        self.num_files = len(self.files)
        self.window_size = window_size
        self.normalize = normalize
        self.differentiate = differentiate
        self.cols = cols
        self.target_cols = target_cols
        self.prediction_size = prediction_size
        self.load_data()

    def load_data(self):
        idx_sum = 0
        for file in self.files:
            data = pd.read_csv(file)[self.cols]
            if type(data) == pd.Series:
                data = data.to_frame()

            dataset = PandasDataset(
                data, self.window_size, self.cols, self.target_cols, self.normalize, self.differentiate, self.prediction_size)
            if self.normalize:
                dataset.normalize()
            if self.differentiate:
                dataset.differentiate()

            # skip importing empty datasets
            try:
                if len(dataset) <= 0:
                    continue
            except:
                continue
            idx_sum += len(dataset)
            self.datasets.append(dataset)
            self.idx_dist.append(idx_sum)

    def __getitem__(self, index: int) -> tuple:
        dataset_idx = np.searchsorted(self.idx_dist, index)
        if dataset_idx > 0:
            index = index - self.idx_dist[dataset_idx-1]
        return self.datasets[dataset_idx][index]

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])

    def collate_fn(self, batch) -> List[torch.Tensor]:
        return [torch.tensor(np.array(x)).float() for x in zip(*batch)]

    @property
    def target_size(self) -> int:
        return len(self.target_cols)
