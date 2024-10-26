import multiprocessing as mp
import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    torch.set_float32_matmul_precision('high')

import pytorch_lightning as lit
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import xgboost as xgb

from data_handling.pandasDataSet import DistributedDataset, PandasDataset
from data_handling.sliceSampler import SliceSampler
from models.GRU import GRU
from models.LSTM import LSTM
from models.LSTM_tower import LSTM_tower
from models.LitModel import LitModel


def torch_train(
        normalize: bool, input_size: int, hidden_size: int,
        num_layers: int, output_size: int, epochs: int, callbacks: list,
        train_dataloader: DataLoader, val_dataloader: DataLoader,
        test_dataloader: DataLoader, model_param: dict) -> LitModel:

    model = LitModel(model_param['class'](
        **model_param['model_args']), lr=model_param['lr'], wd=model_param['wd'])
    trainer = lit.Trainer(max_epochs=epochs, callbacks=callbacks, gradient_clip_val=0.5,
                          gradient_clip_algorithm='norm', deterministic=True)
    trainer.fit(model, train_dataloader, val_dataloader)

    result = trainer.test(
        model, [train_dataloader, val_dataloader, test_dataloader], ckpt_path="best")

    os.makedirs('trained', exist_ok=True)
    filename = f'trained/{model.__class__.__name__}_{input_size}_{hidden_size}_{num_layers}_\
        {output_size}_normalized_{normalize}_testloss_{result[2]["test_loss/dataloader_idx_2"]}.pt'
    torch.save(model.state_dict(), filename)
    return model


def torch_plot(batch_size: int, train_dataset: DistributedDataset, test_dataset: DistributedDataset,
               val_dataset: DistributedDataset, model: LitModel) -> None:

    if torch.cuda.is_available():
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

                batch_x = torch.tensor(batch_x).float()
                batch_y = torch.tensor(batch_y).float()
                if torch.cuda.is_available():
                    batch_x.cuda()
                    batch_y.cuda()
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


def prepare_dataloaders(hyperparams) -> list[DataLoader]:
    train_dataset = DistributedDataset(
        directory=f'data\\{hyperparams["data_prefix"]}train', window_size=30, target_size=hyperparams['target_size'], normalize=hyperparams['normalize'], cols=hyperparams['cols'], target_cols=hyperparams['target_cols'])
    test_dataset = DistributedDataset(
        directory=f'data\\{hyperparams["data_prefix"]}test', window_size=30, target_size=hyperparams['target_size'], normalize=hyperparams['normalize'], cols=hyperparams['cols'], target_cols=hyperparams['target_cols'])
    val_dataset = DistributedDataset(
        directory=f'data\\{hyperparams["data_prefix"]}val', window_size=30, target_size=hyperparams['target_size'], normalize=hyperparams['normalize'], cols=hyperparams['cols'], target_cols=hyperparams['target_cols'])

    train_dataloader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], sampler=SliceSampler(
        train_dataset, slice_size=hyperparams['slice_size'], batch_size=hyperparams['batch_size']), collate_fn=train_dataset.collate_fn, num_workers=hyperparams['num_workers'], persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], sampler=SliceSampler(
        val_dataset, slice_size=hyperparams['slice_size'], batch_size=hyperparams['batch_size']), collate_fn=val_dataset.collate_fn, num_workers=hyperparams['num_workers'], persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], sampler=SliceSampler(
        test_dataset, slice_size=hyperparams['slice_size'], batch_size=hyperparams['batch_size']), collate_fn=test_dataset.collate_fn, num_workers=hyperparams['num_workers'], persistent_workers=True)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    return train_dataloader, val_dataloader, test_dataloader


def initialize_models(hyperparams):
    model_dict = {
        'GRU': {
            'class': GRU,
            'model_args': {
                'input_size': len(hyperparams['cols']),
                'hidden_size': 32,
                'num_layers': 4,
                'output_size': len(hyperparams['target_cols'])
            },
            'lr': hyperparams['lr'],
            'wd': hyperparams['wd']
        },
        'LSTM': {
            'class': LSTM,
            'model_args': {
                'input_size': len(hyperparams['cols']),
                'hidden_size': 32,
                'num_layers': 4,
                'output_size': len(hyperparams['target_cols'])
            },
            'lr': hyperparams['lr'],
            'wd': hyperparams['wd']
        },
        'LSTM_tower': {
            'class': LSTM_tower,
            'model_args': {
                'input_size': len(hyperparams['cols']),
                'output_size': len(hyperparams['target_cols'])
            },
            'lr': hyperparams['lr'],
            'wd': hyperparams['wd']
        }
    }

    return model_dict


def create_callbacks():
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            dirpath='checkpoints',
            filename='model_{model.__class__.__name__}-epoch_{epoch:02d}-loss_{val_loss:.2f}'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )
    ]

    return callbacks


def train(plot_model_performance=False) -> None:
    # set hyperparameters
    hyperparams = {
        'batch_size': 256,
        'slice_size': 64,
        'num_workers': min(mp.cpu_count(), 8),
        'cols': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'target_cols': ['Open', 'High', 'Low', 'Close'],
        'target_size': 1,
        'normalize': True,
        'data_prefix': 'sp500',
        'epochs': 20,
        'lr': 1e-6,
        'wd': 1e-6
    }

    # define callbacks
    callbacks = create_callbacks()

    # define models
    model_dict = initialize_models(hyperparams)

    # create dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(
        hyperparams)

    for model_name, model_param in model_dict.items():
        if issubclass(model_param['class'], nn.Module):
            model = torch_train(
                hyperparams['normalize'], model_param['model_args'],
                model_param['model_args']['hidden_size'],
                model_param['model_args']['num_layers'],
                len(hyperparams['target_cols']
                    ), hyperparams['epochs'], callbacks, train_dataloader,
                val_dataloader, test_dataloader, model_param
            )

            if plot_model_performance:
                torch_plot(hyperparams['batch_size'], train_dataloader.dataset,
                           test_dataloader.dataset, val_dataloader.dataset, model)

        elif issubclass(model_param['class'], xgb.XGBRegressor):
            pass

        else:
            raise ValueError(
                f"Model {model_name} is not a valid model class")

        # free memory
        del model


if __name__ == '__main__':
    train()
