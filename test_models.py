import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import torch
if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    torch.set_float32_matmul_precision('high')

import pytorch_lightning as lit
from models.GRU import GRU
from models.LSTM import LSTM
from models.LSTM_tower import LSTM_tower
from models.LitModel import LitModel
from data_handling.pandasDataSet import PandasDataset
from train import prepare_dataloaders


def evaluate_model(model_name, model_class, hidden_size, num_layers, result_queue, training_params):
    dataloaders = prepare_dataloaders(training_params, sets=['test'])

    try:
        model = model_class(
            input_size=training_params['input_size'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=len(training_params['target_cols'])
        )
    except:
        model = model_class(
            input_size=training_params['input_size'],
            output_size=len(training_params['target_cols'])
        )

    lit_model = LitModel.load_from_checkpoint(
        os.path.join('trained', model_name),
        model=model
    )
    trainer = lit.Trainer(**training_params['trainer_params'])
    mse = trainer.test(lit_model, dataloaders['test'])[0]['test_loss']

    result_queue.put(mse)


def test_models():
    training_params = {
        'batch_size': 512,
        'slice_size': 64,
        'num_workers': min(mp.cpu_count(), 8),
        # 'cols': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'cols': ['Close'],
        'input_size': 1,
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

    model_dict = {
        "GRU": GRU,
        "LSTM": LSTM,
        "LSTM_tower": LSTM_tower,
    }

    columns = [
        'model_name', 'class_name', 'input_size', 'output_size', 'hidden_size', 'num_layers', 'transform', 'mse']

    if not os.path.exists('model_results.csv'):
        with open('model_results.csv', 'w') as f:
            f.write(','.join(columns) + '\n')

    for model_name in os.listdir("trained"):
        split = model_name.split('_')
        if len(split) == 6:
            m_class, num_source, num_target, hidden_size, num_layers, transformation = split
            hidden_size = int(hidden_size)
            num_layers = int(num_layers)
        else:
            m_class = split[0] + '_' + split[1]
            num_source = split[2]
            num_target = split[3]
            transformation = split[4]

        model_class = model_dict[m_class]

        with open('model_results.csv', 'r') as f:
            if model_name in f.read():
                continue

        if transformation == 'stationary':
            training_params['stationary_transform'] = True
            training_params['normalize'] = False
        else:
            training_params['stationary_transform'] = False
            training_params['normalize'] = True

        training_params['create_features'] = False
        if num_source == '1':
            training_params['cols'] = ['Close']
            training_params['input_size'] = 1
        elif num_source == '5':
            training_params['cols'] = [
                'Open', 'High', 'Low', 'Close', 'Volume']
            training_params['input_size'] = 5
        else:
            training_params['cols'] = [
                'Open', 'High', 'Low', 'Close', 'Volume']
            training_params['create_features'] = True
            training_params['input_size'] = 61

        training_params['target_cols'] = ['Close'] if num_target == '1' else [
            'Open', 'High', 'Low', 'Close']

        result_queue = mp.Queue()
        p = mp.Process(target=evaluate_model, args=(
            model_name, model_class, hidden_size, num_layers, result_queue, training_params))
        p.start()
        p.join()
        mse = result_queue.get()

        with open('model_results.csv', 'a') as f:
            f.write(
                f'{model_name},{model_name.split("_")[0]},{len(training_params["cols"])},{len(training_params["target_cols"])},{hidden_size},{num_layers},{"stationary" if training_params["stationary_transform"] else "normalized"},{mse}\n')


def compare_models():
    path = 'to_compare'
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

    model_dict = {
        "GRU": GRU,
        "LSTM": LSTM,
        "LSTM_tower": LSTM_tower,
    }

    for model_name in os.listdir("to_compare"):
        m_class, num_source, num_target, hidden_size, num_layers, transformation = model_name.split(
            '_')
        model_class = model_dict[m_class]
        hidden_size = int(hidden_size)
        num_layers = int(num_layers)
        model_params = model_name.split('_')

        if transformation == 'stationary':
            training_params['stationary_transform'] = True
            training_params['normalize'] = False
        else:
            training_params['stationary_transform'] = False
            training_params['normalize'] = True

        training_params['cols'] = ['Close'] if num_source == '1' else [
            'Open', 'High', 'Low', 'Close', 'Volume']
        training_params['target_cols'] = ['Close'] if num_target == '1' else [
            'Open', 'High', 'Low', 'Close']

        model = model_class(
            input_size=len(training_params['cols']),
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=len(training_params['target_cols'])
        )

        lit_model = LitModel.load_from_checkpoint(
            os.path.join(path, model_name),
            model=model
        )

        data_path = 'data/testy'
        for file in os.listdir(data_path):
            df = pd.read_csv(os.path.join(data_path, file))
            dataset = PandasDataset(df, training_params['cols'], training_params['target_cols'],
                                    1, training_params['normalize'], training_params['stationary_transform'])
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=512, shuffle=False, num_workers=0)

            trainer = lit.Trainer(**training_params['trainer_params'])
            mse = trainer.test(lit_model, dataloader)[0]['test_loss']

            with open('model_results_compare.csv', 'a') as f:
                f.write(f'{model_params[0]},{len(training_params["cols"])},{len(training_params["target_cols"])},{hidden_size},{num_layers},{"stationary" if training_params["stationary_transform"] else "normalized"},{file.split(".")[1]},{mse}\n')


def create_features(df: pd.DataFrame) -> None:
    from feature_creation import indicators
    indicators.add_candlestick_patterns(df)
    indicators.add_candlestick_patterns(df)
    indicators.add_moving_averages(df)


def evaluate_best_models():
    best_path = 'best_ml_models'
    # load models
    gru = GRU(input_size=1, hidden_size=128, output_size=1, num_layers=4)
    lit_gru = LitModel.load_from_checkpoint(
        os.path.join(best_path, 'GRU_1_1_128_4_stationary'),
        model=gru
    )
    lstm_tower = LSTM_tower(input_size=61, output_size=4)
    lit_lstm_tower = LitModel.load_from_checkpoint(
        os.path.join(best_path, 'LSTM_tower_61_4_stationary'),
        model=lstm_tower
    )

    error_dict = {}
    data_path = 'data_ohlcv'

    if os.path.exists('results'):
        for file in os.listdir('results'):
            os.remove(os.path.join('results', file))
    else:
        os.mkdir('results')

    for file in os.listdir(data_path):
        df = pd.read_csv(os.path.join(data_path, file))[['Close']]
        dataset_gru = PandasDataset(
            df,
            window_size=30,
            cols=['Close'],
            target_cols=['Close'],
            stationary_tranform=True
        )
        dataset_gru.apply_transform()
        dataset_gru.dataframe = dataset_gru.dataframe[int(0.7*len(df)):]
        orig_df = pd.read_csv(os.path.join(data_path, file))[['Close']]
        orig_dset_gru = PandasDataset(
            orig_df,
            window_size=30,
            cols=['Close'],
            target_cols=['Close']
        )
        orig_dset_gru.dataframe = orig_dset_gru.dataframe[-len(dataset_gru.dataframe):]


        df = pd.read_csv(os.path.join(data_path, file))[['Open', 'High', 'Low', 'Close', 'Volume']]
        dataset_lstm_tower = PandasDataset(
            df,
            window_size=30,
            cols=['Open', 'High', 'Low', 'Close', 'Volume'],
            target_cols=['Open', 'High', 'Low', 'Close'],
            stationary_tranform=True
        )
        create_features(dataset_lstm_tower.dataframe)
        dataset_lstm_tower.apply_transform()
        dataset_lstm_tower.columns = dataset_lstm_tower.dataframe.columns.tolist()
        dataset_lstm_tower.dataframe = dataset_lstm_tower.dataframe[int(0.7*len(df)):]
        orig_df = pd.read_csv(os.path.join(data_path, file))[['Open', 'High', 'Low', 'Close', 'Volume']]
        orig_dset_lstm = PandasDataset(
            orig_df,
            window_size=30,
            cols=['Open', 'High', 'Low', 'Close', 'Volume'],
            target_cols=['Open', 'High', 'Low', 'Close'],
            stationary_tranform=True
        )
        orig_dset_lstm.dataframe = orig_dset_lstm.dataframe[-len(dataset_lstm_tower.dataframe):]

        gru_y, gru_pred = [], []
        for idx in range(len(dataset_gru)):
            x, y = dataset_gru[idx]
            x_source, y_source = orig_dset_gru[idx]
            x_source = x_source[-1]
            y_hat = lit_gru(torch.tensor(x).unsqueeze(0).float()).cpu().detach().numpy()
            y_pred = x_source + x_source * y_hat
            gru_y.append(y_source)
            gru_pred.append(y_pred)
        gru_mse = np.mean((np.array(gru_y) - np.array(gru_pred))**2)
        error_dict[f'gru_{file.split(".")[0]}'] = gru_mse

        with open(f'results\\results_gru_{file.split(".")[0]}.csv', 'w') as f:
            f.write('y,pred\n')
            for y, pred in zip(gru_y, gru_pred):
                f.write(f'{y},{pred}\n')

        lstm_tower_y, lstm_tower_y_pred = [], []
        for idx in range(len(dataset_lstm_tower)):
            x, y = dataset_lstm_tower[idx]
            x_source, y_source = orig_dset_lstm[idx]
            x_source = x_source[-1][-2]
            y_hat = lit_lstm_tower(torch.tensor(x).unsqueeze(0).float()).cpu().detach().numpy()
            y_pred = x_source + x_source * y_hat[0][-1]
            lstm_tower_y.append(y_source[0][-1])
            lstm_tower_y_pred.append(y_pred)
        lstm_tower_mse = np.mean((np.array(lstm_tower_y) - np.array(lstm_tower_y_pred))**2)
        error_dict[f'lstm_tower_{file.split(".")[0]}'] = lstm_tower_mse

        with open(f'results\\results_lstm_tower_{file.split(".")[0]}.csv', 'w') as f:
            f.write('y,pred\n')
            for y, pred in zip(lstm_tower_y, lstm_tower_y_pred):
                f.write(f'{y},{pred}\n')

    err_df = pd.DataFrame.from_dict(error_dict, orient='index', columns=['mse'])
    err_df.to_csv('results\\errors.csv')


if __name__ == '__main__':
    evaluate_best_models()
    # test_models()
    # compare_models()
