import multiprocessing as mp
import os
from typing import List, Dict
import shutil

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

from sklearn.metrics import mean_squared_error

from data_handling.pandasDataSet import DistributedDataset, PandasDataset
from data_handling.sliceSampler import SliceSampler
from models.GRU import GRU
from models.LSTM import LSTM
from models.LSTM_tower import LSTM_tower
from models.LitModel import LitModel
from models.ARIMA import ARIMA


def torch_train(model_params: Dict, training_params: Dict, callbacks: List[lit.Callback], dataloaders: Dict) -> LitModel:
    model = LitModel(
        model_params['class'](**model_params['model_args']),
        lr=model_params['lr'],
        wd=model_params['wd']
    )
    trainer = lit.Trainer(
        callbacks=callbacks,
        **training_params['trainer_params']
    )
    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    result = trainer.test(model, list(dataloaders.values()), ckpt_path="best")

    os.makedirs('trained', exist_ok=True)
    filename = ''.join([
        f'trained/{model.model.__class__.__name__}_',
        f'_'.join(str(x) for x in list(model_params['model_args'].values())),
        f'_tloss-{result[0]["test_loss/dataloader_idx_0"]}',
        '.ckpt'
    ])

    checkpoint_callback = next(
        x for x in callbacks if isinstance(x, ModelCheckpoint))
    shutil.copy(checkpoint_callback.best_model_path, filename)
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


def prepare_dataloaders(hyperparams: Dict, window_size=30) -> Dict[str, DataLoader]:
    dataloaders = {}
    for split in ['train', 'test', 'val']:
        dset = DistributedDataset(
            directory=f'data/{hyperparams["data_prefix"]}{split}',
            window_size=window_size,
            normalize=hyperparams['normalize'],
            cols=hyperparams['cols'],
            target_cols=hyperparams['target_cols'],
            prediction_size=hyperparams['prediction_size']
        )

        dataloaders[split] = DataLoader(
            dset,
            batch_size=hyperparams['batch_size'],
            sampler=SliceSampler(
                dset,
                slice_size=hyperparams['slice_size'],
                batch_size=hyperparams['batch_size']),
            collate_fn=dset.collate_fn,
            num_workers=hyperparams['num_workers'],
            persistent_workers=True if hyperparams['num_workers'] > 0 else False
        )
    return dataloaders


def initialize_models(hyperparams: Dict) -> Dict:
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
        },
        'ARIMA': {
            'class': ARIMA,
            'model_args': {
                'p':1,
                'd':1,
                'q':1
            }
        }
    }
    return model_dict


def create_callbacks() -> List[lit.Callback]:
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            dirpath='checkpoints',
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )
    ]
    return callbacks


def train(plot_model_performance=False, model_dict=None) -> None:
    # set training parameters
    training_params = {
        'batch_size': 256,
        'slice_size': 64,
        'num_workers': min(mp.cpu_count(), 8),
        'cols': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'target_cols': ['Open', 'High', 'Low', 'Close'],
        'prediction_size': 1,
        'normalize': True,
        'differentiate': True,
        'data_prefix': 'sp500',
        'lr': 1e-6,
        'wd': 1e-6,
        'trainer_params': {
            'gradient_clip_val': 0.5,
            'gradient_clip_algorithm': 'norm',
            'deterministic': True,
            'max_epochs': 2
        }
    }

    # define models
    model_dict = initialize_models(
        training_params) if model_dict is None else model_dict

    # create dataloaders
    dataloaders = prepare_dataloaders(training_params)

    for model_name, model_params in model_dict.items():
        if issubclass(model_params['class'], nn.Module):
            model = torch_train(
                model_params, training_params, create_callbacks(), dataloaders)
            if plot_model_performance:
                torch_plot(training_params, dataloaders, model)

        elif issubclass(model_params['class'], ARIMA):
            # TO DO
            # train_price = data['Close'][:train_size] 
            # test_price = data['Close'][train_size:]
            
            # arima_model = model_params['class'](**model_params['model_args'])
            # arima_model.fit(train_price)
            # predictions = arima_model.roll_forecast(test_price)
            
            # mse = round(mean_squared_error(test_price, predictions), 2)
            # print(f'MSE: {mse}')
            pass
            
        elif issubclass(model_params['class'], a):
            pass

        else:
            raise ValueError(
                f"Model {model_name} is not a valid model class")

        # free memory
        del model


if __name__ == '__main__':
    train()
