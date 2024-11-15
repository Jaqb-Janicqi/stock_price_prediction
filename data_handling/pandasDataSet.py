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
    def __init__(self, dataframe: pd.DataFrame, window_size: int, 
                 cols=['Close'], target_cols=['Close'], normalize=False, 
                 prediction_size=1, drop_null_rows=True):
        self._dataframe = dataframe
        self._window_size = window_size
        self._cols = cols
        self._target_cols = target_cols
        self._should_normalize = normalize
        self._prediction_size = prediction_size
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._drop_null_rows = drop_null_rows
        if self._should_normalize:
            self.normalize()
        if self._drop_null_rows:
            self._dataframe.dropna(axis=0)

    def __getitem__(self, index: int) -> tuple:
        # select cols for x, and target_cols for y
        x = self._dataframe.iloc[index:index+self._window_size][self._cols].values
        y = self._dataframe.iloc[index+self._window_size:index +
                    self._window_size+self._prediction_size][self._target_cols].values        
        return np.array(x), np.array(y)

    def __len__(self) -> int:
        return len(self._dataframe) - self._window_size - self._prediction_size

    def collate_fn(self, batch) -> List[torch.Tensor]:
        return [torch.tensor(np.array(x)).float() for x in zip(*batch)]

    def normalize(self):
        # self.dataframe = (self.dataframe - self.dataframe.mean()) / self.dataframe.std()
        self._dataframe[self._cols] = self.scaler.fit_transform(self._dataframe[self._cols])

    def denormalize(self, data):
        return self.scaler.inverse_transform(data)
    
    def cast(self, columns: List[str], dtypes: List[str]):
        self._dataframe[columns] = self._dataframe[columns].astype(dtypes)

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
    def __init__(self, directory: str, window_size: int, normalize: bool = False, 
                 cols=['Close'], target_cols=['Close'], 
                 prediction_size=1, create_features=True):
        self._datasets = []
        self._idx_dist = []
        self._used_indices = []
        self._files = list_files(directory)
        self._num_files = len(self._files)
        self._window_size = window_size
        self._normalize = normalize
        self._cols = cols
        self._target_cols = target_cols
        self._prediction_size = prediction_size
        self._should_create_features = create_features
        self._load_data()

    def _create_features(self, df: pd.DataFrame) -> None:
        from feature_creation import indicators
        indicators.add_candlestick_patterns(df)
        indicators.add_candlestick_patterns(df)
        indicators.add_moving_averages(df)

    def _load_data(self):
        idx_sum = 0
        for file in self._files:
            data = pd.read_csv(file)[self._cols]
            if type(data) == pd.Series:
                data = data.to_frame()
            if self._should_create_features:
                self._create_features(data)

            dataset = PandasDataset(
                data, self._window_size, data.columns.tolist(), self._target_cols, self._normalize, self._prediction_size)
            if self._normalize:
                dataset.normalize()

            # skip importing empty datasets
            try:
                if len(dataset) <= 0:
                    continue
            except:
                continue
            idx_sum += len(dataset)
            self._datasets.append(dataset)
            self._idx_dist.append(idx_sum)
        self._used_indices = list(range(idx_sum))

    def __getitem__(self, index: int) -> tuple:
        if index < 0 or index >= len(self):
            raise IndexError
        
        # map index with used indices
        index = self._used_indices[index]

        # find dataset index
        dataset_idx = np.searchsorted(self._idx_dist, index)
        if dataset_idx > 0:
            index = index - self._idx_dist[dataset_idx-1]
        return self._datasets[dataset_idx][index]

    def __len__(self) -> int:
        return len(self._used_indices)

    def collate_fn(self, batch) -> List[torch.Tensor]:
        return [torch.tensor(np.array(x)).float() for x in zip(*batch)]

    @property
    def target_size(self) -> int:
        return len(self._target_cols)
    
    @property
    def used_indices(self):
        return self._used_indices
    
    @used_indices.setter
    def used_indices(self, indices: List[int]):
        self._used_indices = indices

    def reset_indices(self):
        self._used_indices = list(range(sum([len(dataset) for dataset in self._datasets])))

    def cast(self, columns: List[str], dtypes: List[str]):
        for dataset in self._datasets:
            dataset.cast(columns, dtypes)
