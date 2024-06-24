import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as lit
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data_handling.pandasDataSet import DistributedDataset
from data_handling.sliceSampler import SliceSampler
from models.GRU import LitGRU
from models.LSTM import LitLSTM

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    # set hyperparameters
    batch_size = 256
    slice_size = 64
    num_workers = 8
    cols = ['Close', 'Volume']
    target_cols = ['Close']
    normalize = False

    # model params
    input_size = len(cols)
    hidden_size = 512
    num_layers = 4
    output_size = len(target_cols)

    # training params
    epochs = 3
    lr = 1e-4
    wd = 1e-6

    # define callbacks
    callbacks = [
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3),
        EarlyStopping(monitor='val_loss', patience=5, mode='min')
    ]

    # prefix = 'sp500'
    prefix = 'btc_'

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
    model = LitLSTM(input_size, hidden_size, num_layers,
                    output_size, lr=lr, wd=wd).cuda()
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
