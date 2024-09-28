import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as lit
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data_handling.pandasDataSet import DistributedDataset, PandasDataset
from data_handling.sliceSampler import SliceSampler
from models.GRU import LitGRU
from models.LSTM import LitLSTM
from models.LSTM_tower import LitLSTMtower

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    # set hyperparameters
    batch_size = 256
    slice_size = 64
    num_workers = 8
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_cols = ['Close']
    normalize = True

    # model params
    input_size = len(cols)
    hidden_size = 32
    num_layers = 4
    output_size = len(target_cols)

    # training params
    epochs = 20
    lr = 1e-6
    wd = 1e-6

    # define callbacks
    callbacks = [
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3),
        EarlyStopping(monitor='val_loss', patience=5, mode='min')
    ]

    # prefix = 'sp500'
    # prefix = 'btc_'
    prefix = 'eurusd_15_'

    # create datasets
    train_dataset = DistributedDataset(
        directory=f'data\\{prefix}train', window_size=30, target_size=1, normalize=normalize, cols=cols, target_cols=target_cols)
    test_dataset = DistributedDataset(
        directory=f'data\\{prefix}test', window_size=30, target_size=1, normalize=normalize, cols=cols, target_cols=target_cols)
    val_dataset = DistributedDataset(
        directory=f'data\\{prefix}val', window_size=30, target_size=1, normalize=normalize, cols=cols, target_cols=target_cols)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=SliceSampler(
        train_dataset, slice_size=slice_size, batch_size=batch_size), collate_fn=train_dataset.collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=SliceSampler(
        val_dataset, slice_size=slice_size, batch_size=batch_size), collate_fn=val_dataset.collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=SliceSampler(
        test_dataset, slice_size=slice_size, batch_size=batch_size), collate_fn=test_dataset.collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    print(f"Train dataset length: {len(train_dataset)}")

    # create a model
    # model = LitGRU(input_size, hidden_size, num_layers,
    #                output_size, lr=lr, wd=wd).cuda()
    # model = LitLSTM(input_size, hidden_size, num_layers,
    #                 output_size, lr=lr, wd=wd).cuda()
    model = LitLSTMtower(input_size, output_size, lr=lr, wd=wd).cuda()
    # create a trainer
    trainer = lit.Trainer(max_epochs=epochs, callbacks=callbacks, gradient_clip_val=0.5,
                          gradient_clip_algorithm='norm', deterministic=True)
    # train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    # test the model
    result = trainer.test(
        model, [train_dataloader, val_dataloader, test_dataloader], ckpt_path="best")
    print(result)

    # save the model with class name and model params
    if os.path.exists('trained') == False:
        os.makedirs('trained')
    filename = f'trained\\{model.__class__.__name__}_{input_size}_{hidden_size}_{num_layers}_{output_size}_normalized_{normalize}_testloss_{result[2]["test_loss/dataloader_idx_2"]}.pt'
    torch.save(model.state_dict(), filename)

    # # test the model on AAPL
    # train_dataset = PandasDataset(pd.read_csv(f'data\\sp500train\\AAPL_1h.csv')[
    #                               cols], window_size=30, target_size=1, cols=cols, target_cols=target_cols, normalize=normalize)
    # test_dataset = PandasDataset(pd.read_csv(f'data\\sp500test\\AAPL_1h.csv')[
    #                              cols], window_size=30, target_size=1, cols=cols, target_cols=target_cols, normalize=normalize)
    # val_dataset = PandasDataset(pd.read_csv(f'data\\sp500val\\AAPL_1h.csv')[
    #                             cols], window_size=30, target_size=1, cols=cols, target_cols=target_cols, normalize=normalize)

    # # create dataloaders
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=SliceSampler(
    #     train_dataset, slice_size=slice_size, batch_size=batch_size), collate_fn=train_dataset.collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=SliceSampler(
    #     val_dataset, slice_size=slice_size, batch_size=batch_size), collate_fn=val_dataset.collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=SliceSampler(
    #     test_dataset, slice_size=slice_size, batch_size=batch_size), collate_fn=test_dataset.collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    model.cuda()
    model.eval()
    dsets = [train_dataset, val_dataset, test_dataset]
    with torch.no_grad():
        for dset in dsets:
            maxlen = 1000 if len(dset) > 1000 else len(dset)
            x = []
            y = []
            yhat = []
            for i in range(0, maxlen, batch_size):
                batch_x = []
                batch_y = []
                for j in range(i, min(i + batch_size, len(dset))):
                    x_, y_ = dset[j]
                    batch_x.append(x_)
                    batch_y.append(y_)
                batch_x = torch.tensor(batch_x).float().cuda()
                batch_y = torch.tensor(batch_y).float().cuda()
                batch_y_hat: torch.Tensor = model(batch_x)
                x.append(batch_x)
                y.append(batch_y)
                yhat.append(batch_y_hat.cpu())
            x = torch.cat(x)
            y = torch.cat(y)
            y_hat = torch.cat(yhat)
            plt.plot(y.squeeze().cpu().numpy(), label='y')
            plt.plot(y_hat.squeeze().cpu().numpy(), label='y_hat')
            plt.legend()
            plt.show()
