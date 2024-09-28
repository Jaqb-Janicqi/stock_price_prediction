import numpy as np
import torch.nn as nn


class ModelClass:
    def __init__(self, model) -> None:
        self._model = model

    def predict(self, x) -> np.ndarray:
        y = self._model.predict(x)
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y)
            except:
                raise ValueError(
                    f"Model output is not a numpy array and cannot be converted to one: {y}")
        return y

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def model_name(self) -> str:
        return self._model.__class__.__name__
