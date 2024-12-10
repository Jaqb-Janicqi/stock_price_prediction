import numpy as np
import pytorch_lightning as lit
import torch
import torch.nn as nn
import torch.optim as optim


class LitModel(lit.LightningModule):
    def __init__(self, model, lr=1e-2, wd=1e-6, loss_function=nn.MSELoss(), save_hparams=False) -> None:
        super(LitModel, self).__init__()
        self._model_instance = model
        self._loss_function = loss_function
        self._lr = lr
        self._wd = wd
        if save_hparams:
            self.save_hyperparameters()
        self._transformation = None

    def forward(self, x) -> torch.Tensor:
        return self._model_instance(x)

    def predict(self, x) -> np.ndarray:
        # for compatibility with other frameworks
        return self.forward(x).to('cpu').detach().numpy()

    def configure_optimizers(self, optimizer=None, scheduler=None) -> dict:
        if optimizer is None:
            optimizer = optim.AdamW(
                self.parameters(), lr=self._lr, weight_decay=self._wd, amsgrad=True)
        if scheduler is None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-8)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def _step(self, batch) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        return self._loss_function(y_hat, y.squeeze(1))

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._step(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        loss = self._step(batch)
        self.log('test_loss', loss)
        return loss

    @property
    def model(self) -> nn.Module:
        return self._model_instance

    @property
    def loss_function(self) -> nn.Module:
        return self._loss_function

    @property
    def lr(self) -> float:
        return self._lr

    @property
    def wd(self) -> float:
        return self._wd

    @lr.setter
    def lr(self, value) -> None:
        self._lr = value

    @wd.setter
    def wd(self, value) -> None:
        self._wd = value

    @property
    def input_size(self) -> int:
        return self._model_instance.input_size
    
    @property
    def output_size(self) -> int:
        return self._model_instance.output_size
    
    @property
    def transformation(self) -> str:
        return self._transformation
    
    @transformation.setter
    def transformation(self, value) -> None:
        self._transformation = value