import torch
import torch.nn as nn

class ModelClass:
    def __init__(self, model) -> None:
        self._model = model

    def predict(self, x) -> torch.Tensor:
        return self._model.predict(x)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def model_name(self) -> str:
        return self._model.__class__.__name__
