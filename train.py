import multiprocessing as mp
import os
import pickle
from typing import List, Dict
import shutil

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
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
from models.RidgeRegression import RidgeRegression


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
    result = trainer.test(model, dataloaders['test'], ckpt_path="best")

    os.makedirs('trained', exist_ok=True)
    filename = ''.join([
        f'trained/{model.model.__class__.__name__}_',
        f'_'.join(str(x) for x in list(model_params['model_args'].values())),
        f'_tloss-{result[0]["test_loss"]:.5f}',
        '.ckpt'
    ])

    checkpoint_callback = next(
        x for x in callbacks if isinstance(x, ModelCheckpoint))
    shutil.copy(checkpoint_callback.best_model_path, filename)


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


def prepare_dataloaders(hyperparams: Dict, window_size=30, sets=['train', 'test', 'val']) -> Dict[str, DataLoader]:
    dataloaders = {}
    for split in sets:
        dset = DistributedDataset(
            directory=f'data/{hyperparams["data_prefix"]}{split}',
            window_size=window_size,
            normalize=hyperparams['normalize'],
            cols=hyperparams['cols'],
            target_cols=hyperparams['target_cols'],
            prediction_size=hyperparams['prediction_size'],
            create_features=hyperparams['create_features'],
            stationary_transform=hyperparams['stationary_transform']
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
        'LSTM_tower': {
            'class': LSTM_tower,
            'model_args': {
                'input_size': 1,
                'output_size': len(hyperparams['target_cols'])
            },
            'lr': hyperparams['lr'],
            'wd': hyperparams['wd']
        },
        # 'GRU': {
        #     'class': GRU,
        #     'model_args': {
        #         'input_size': 5,
        #         'hidden_size': 512,
        #         'num_layers': 8,
        #         'output_size': len(hyperparams['target_cols'])
        #     },
        #     'lr': hyperparams['lr'],
        #     'wd': hyperparams['wd']
        # },
        # 'LSTM': {
        #     'class': LSTM,
        #     'model_args': {
        #         'input_size': 5,
        #         'hidden_size': 512,
        #         'num_layers': 8,
        #         'output_size': len(hyperparams['target_cols'])
        #     },
        #     'lr': hyperparams['lr'],
        #     'wd': hyperparams['wd']
        # },
        'GRU': {
            'class': GRU,
            'model_args': {
                'input_size': 1,
                'hidden_size': 256,
                'num_layers': 8,
                'output_size': len(hyperparams['target_cols'])
            },
            'lr': hyperparams['lr'],
            'wd': hyperparams['wd']
        },
        'LSTM': {
            'class': LSTM,
            'model_args': {
                'input_size': 1,
                'hidden_size': 256,
                'num_layers': 8,
                'output_size': len(hyperparams['target_cols'])
            },
            'lr': hyperparams['lr'],
            'wd': hyperparams['wd']
        },
        'GRU_shallow': {
            'class': GRU,
            'model_args': {
                'input_size': 1,
                'hidden_size': 256,
                'num_layers': 4,
                'output_size': len(hyperparams['target_cols'])
            },
            'lr': hyperparams['lr'],
            'wd': hyperparams['wd']
        },
        'LSTM_shallow': {
            'class': LSTM,
            'model_args': {
                'input_size': 1,
                'hidden_size': 256,
                'num_layers': 4,
                'output_size': len(hyperparams['target_cols'])
            },
            'lr': hyperparams['lr'],
            'wd': hyperparams['wd']
        },
        'GRU_small': {
            'class': GRU,
            'model_args': {
                'input_size': 1,
                'hidden_size': 128,
                'num_layers': 4,
                'output_size': len(hyperparams['target_cols'])
            },
            'lr': hyperparams['lr'],
            'wd': hyperparams['wd']
        },
        'LSTM_small': {
            'class': LSTM,
            'model_args': {
                'input_size': 1,
                'hidden_size': 128,
                'num_layers': 4,
                'output_size': len(hyperparams['target_cols'])
            },
            'lr': hyperparams['lr'],
            'wd': hyperparams['wd']
        },
        # 'ARIMA': {
        #     'class': ARIMA,
        #     'model_args': {
        #         'p':1,
        #         'd':1,
        #         'q':1
        #     }
        # },
        # 'RidgeRegression':{
        #     'class': RidgeRegression,
        #     'model_args': {
        #         'alpha': 1.0
        #     }
        # }
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
        'batch_size': 512,
        'slice_size': 64,
        'num_workers': min(mp.cpu_count(), 8),
        # 'cols': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'cols': ['Close'],
        # 'target_cols': ['Open', 'High', 'Low', 'Close'],
        'target_cols': ['Close'],
        'prediction_size': 1,
        'normalize': True,
        'data_prefix': 'sp500',
        'lr': 1e-3,
        'wd': 1e-6,
        'create_features': False,
        'stationary_transform': False,
        'trainer_params': {
            'deterministic': True,
            'max_epochs': 50
        }
    }

    # define models
    model_dict = initialize_models(
        training_params) if model_dict is None else model_dict

    # create dataloaders
    dataloaders = prepare_dataloaders(training_params)

    for model_name, model_params in model_dict.items():
        if issubclass(model_params['class'], nn.Module):
            p = mp.Process(target=torch_train, args=(
                model_params, training_params, create_callbacks(), dataloaders))
            p.start()
            p.join()


        elif issubclass(model_params['class'], ARIMA):
            # Load training and testing data
            data_train = DistributedDataset(
                directory=f'data/sp500train',
                window_size=1,
                normalize=False,
                cols=['Datetime', 'Close'],
                target_cols=['Close'],
                prediction_size=1,
                create_features=False
            )
            data_test = DistributedDataset(
                directory=f'data/sp500test',
                window_size=1,
                normalize=False,
                cols=['Datetime', 'Close'],
                target_cols=['Close'],
                prediction_size=1,
                create_features=False
            )

            for p_dataset in data_train.datasets:
                p_dataset.dataframe['Datetime'] = pd.to_datetime(
                    p_dataset.dataframe['Datetime'], utc=True)
                p_dataset.dataframe.set_index('Datetime', inplace=True)
                p_dataset.columns = ['Close']

            for p_dataset in data_test.datasets:
                p_dataset.dataframe['Datetime'] = pd.to_datetime(
                    p_dataset.dataframe['Datetime'], utc=True)
                p_dataset.dataframe.set_index('Datetime', inplace=True)
                p_dataset.columns = ['Close']

            # Initialize and fit the model
            arima_model = ARIMA(p=1, d=1, q=1)

            # Fit the model on training data
            arima_model.fit(data_train)

            # Make predictions on the test set
            predictions = arima_model.rolling_forecast(data_test)

            # Calculate Mean Squared Error
            mse = round(mean_squared_error(test_price, predictions), 2)
            print(f'MSE: {mse}')
            # mape = round(np.mean(np.abs((test_price - predictions)/test_price))*100, 2)
            # print(f'MSE: {mape}')

            # Plotting the results
            plt.plot(data.index[:train_size], train_price,
                     color="blue", label="Train")
            plt.plot(data.index[train_size:], test_price,
                     color="grey", label="Test")
            plt.plot(data.index[train_size:], predictions,
                     label="Predicted", color='red', ls=':')
            plt.title("ARIMA Model Prediction")
            plt.legend()
            plt.show()
            
            os.makedirs('trained', exist_ok=True)

            filename = os.path.join(
                'trained',
                f'ARIMA_{arima_model.p}-{arima_model.d}-{arima_model.q}_mse-{mse:.5f}.ckpt'
            )

            print(f'Saving model to {filename}')

            try:
                with open(filename, 'wb') as file:
                    pickle.dump(arima_model, file)
                print(f"Model successfully saved to {filename}")
            except Exception as e:
                print(f"Error saving the model: {e}")

        elif issubclass(model_params['class'] == RidgeRegression):
            #!!! Loading, splitting data go outside the model class !!!
            # as well as plotting and calculating metrics
            data = DistributedDataset(
                directory=f'data/sp500',
                window_size=1,
                normalize=True,
                cols=['Close'],
                target_cols=['Close'],
                prediction_size=1,
                create_features=True
            )
            idx_max = len(data)
            idx_dist = np.arange(idx_max, dtype=int)
            idx_test = np.random.choice(idx_dist, len(data)/3)
            dset_test = DistributedDataset(
                directory=f'data/sp500',
                window_size=1,
                normalize=True,
                cols=['Close'],
                target_cols=['Close'],
                prediction_size=1,
                create_features=False
            )
            dset_test.used_indices = idx_test
            dset_train = data
            dset_train.used_indices = [
                idx for idx in idx_dist if idx not in idx_test]

            # Load training and testing data
            # data = pd.read_csv('data/sp500/AAPL_1h.csv', low_memory=False)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data.set_index('Datetime', inplace=True)

            # Calculate the Daily Percentage Change in stock price - returns
            data['Returns'] = (data['Close'].pct_change() * 100).round(2)
            data.dropna(inplace=True)
            data.head()

            # For 0.7 train and 0.3 test split model provides better results than 0.8 train and 0.2 test split
            train_size = int(len(data)*0.7)
            train_price = dset_train['Close']
            test_price = dset_test['Close']

            # Initialize and fit the model
            ridge_model = RidgeRegression(alpha=1)
            ridge_model.history = list(data['Returns'][0:train_size])

            predicted_returns = ridge_model.rolling_forecast(
                data['Returns'][train_size:])

            predicted_prices = []
            for i, predicted_return in enumerate(predicted_returns):
                new_price = test_price.iloc[i -
                                            1] if i > 0 else train_price.iloc[-1]
                predicted_prices.append(
                    new_price * (1 + predicted_return / 100))

            # Calculate Mean Squared Error
            mse = round(mean_squared_error(test_price, predicted_prices), 2)
            print(f'MSE: {mse}')

            # Plot the results
            plt.plot(data.index[:train_size], train_price,
                     color="blue", label="Train")
            plt.plot(data.index[train_size:], test_price,
                     color="grey", label="Test")
            plt.plot(data.index[train_size:], predicted_prices,
                     label="Predicted", color='red', ls=':')
            plt.title("Ridge Regression Model Prediction")
            plt.legend()
            plt.show()
            
            os.makedirs('trained', exist_ok=True)

            filename = os.path.join(
                'trained',
                f'RidgeRegression_{ridge_model.alpha}_mse-{mse:.5f}.ckpt'
            )

            print(f'Saving model to {filename}')

            try:
                with open(filename, 'wb') as file:
                    pickle.dump(ridge_model, file)
                print(f"Model successfully saved to {filename}")
            except Exception as e:
                print(f"Error saving the model: {e}")
                
        # elif issubclass(model_params['class'], a):
        #     pass

        else:
            raise ValueError(
                f"Model {model_name} is not a valid model class")


if __name__ == '__main__':
    train()
