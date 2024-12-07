import multiprocessing as mp
import os

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
            input_size=len(training_params['cols']),
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
        try:
            m_class, num_source, num_target, hidden_size, num_layers, transformation = model_name.split(
                '_')
            hidden_size = int(hidden_size)
            num_layers = int(num_layers)
        except:
            m_class, num_source, num_target, transformation = model_name.split(
                '_')
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
            training_params['cols'] = ['Open', 'High', 'Low', 'Close', 'Volume']
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


if __name__ == '__main__':
    test_models()
    # compare_models()
