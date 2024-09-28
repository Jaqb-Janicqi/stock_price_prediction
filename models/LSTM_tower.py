import pytorch_lightning as lit
import torch.nn as nn
import torch.optim as optim


class lstm_tower_model(nn.Module):
    """
    Analysis of Bitcoin Price Prediction Using Machine Learning - Junwei Chen
    """
    def __init__(self, input_size, output_size):
        super(lstm_tower_model, self).__init__()
        # self.lstm_tower = nn.Sequential(
        #     nn.LSTM(input_size, 128, 1, batch_first=True, dropout=0.2),
        #     nn.ReLU(),
        #     nn.LSTM(128, 128, 1, batch_first=True, dropout=0.3),
        #     nn.ReLU(),
        #     nn.LSTM(128, 256, 1, batch_first=True, dropout=0.4),
        #     nn.ReLU(),
        #     nn.LSTM(256, 256, 1, batch_first=True, dropout=0.5),
        #     nn.Linear(256, output_size)
        # )
        self.lstm_tower = nn.ParameterList()
        self.lstm_tower.append(nn.LSTM(input_size, 128, 1, batch_first=True))
        self.lstm_tower.append(nn.Dropout(0.1))
        self.lstm_tower.append(nn.ReLU())
        self.lstm_tower.append(nn.LSTM(128, 128, 1, batch_first=True))
        self.lstm_tower.append(nn.Dropout(0.1))
        self.lstm_tower.append(nn.ReLU())
        self.lstm_tower.append(nn.LSTM(128, 256, 1, batch_first=True))
        self.lstm_tower.append(nn.Dropout(0.1))
        self.lstm_tower.append(nn.ReLU())
        self.lstm_tower.append(nn.LSTM(256, 256, 1, batch_first=True))
        self.lstm_tower.append(nn.Linear(256, output_size))

    def forward(self, x):
        for layer in self.lstm_tower:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            else:
                x = layer(x)
        # Use only the output of the last time step
        return x[:, -1, :]


class LitLSTMtower(lit.LightningModule):
    def __init__(self, input_size, output_size, lr=1e-2, wd=1e-6):
        super(LitLSTMtower, self).__init__()
        self.model = lstm_tower_model(input_size, output_size)
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.wd = wd

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd, amsgrad=False)
        reducer = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-10)
        return {'optimizer': optimizer, 'lr_scheduler': reducer, 'monitor': 'val_loss'}

    def step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        return self.loss_fn(y_hat, y.squeeze(1))

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.step(batch)
        self.log('test_loss', loss)
        return loss
