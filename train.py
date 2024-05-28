import torch
from torch.utils.data import DataLoader
import pytorch_lightning as lit
from data_handling.pandasDataSet import DistributedDataset
from data_handling.sliceSampler import SliceSampler
from models.GRU import LitGRU

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    train_dataset = DistributedDataset(
        directory='data\\sp500train', window_size=30, target_size=1, normalize=True)
    test_dataset = DistributedDataset(
        directory='data\\sp500test', window_size=30, target_size=1, normalize=True)
    val_dataset = DistributedDataset(
        directory='data\\sp500val', window_size=30, target_size=1, normalize=True)

    batch_size = 256
    slice_size = 32
    num_workers = 4

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=SliceSampler(
        train_dataset, slice_size=slice_size, batch_size=batch_size), collate_fn=train_dataset.collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=SliceSampler(
        val_dataset, slice_size=slice_size, batch_size=batch_size), collate_fn=val_dataset.collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=SliceSampler(
        test_dataset, slice_size=slice_size, batch_size=batch_size), collate_fn=test_dataset.collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # create a model
    model = LitGRU(input_size=1, hidden_size=512,
                   num_layers=4, output_size=1).cuda()
    # create a trainer
    trainer = lit.Trainer(max_epochs=2)
    # train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    # test the model
    result = trainer.test(model, test_dataloader)
    print(result)
