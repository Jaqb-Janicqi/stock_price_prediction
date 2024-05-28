import pytorch_lightning as lit
import torch.nn as nn
import torch.optim as optim


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        # Use only the output of the last time step
        out = self.fc(out[:, -1, :])
        return out


class LitGRU(lit.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=1e-2, wd=1e-6):
        super(LitGRU, self).__init__()
        self.model = GRU(input_size, hidden_size, num_layers, output_size)
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.wd = wd

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd, amsgrad=True)
        reducer = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-8)
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

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('test_loss', loss)
        return loss
