import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader


def dataprocess(path:str, val_rate=0.1, test_rate=None, batch_size=32, seed=666):
    r"""
    Pre-process the data and return the dataloader.
    
    Args:
    ---
        path: str, data path
        val_rate: float (Default: 0.1)
        test_rate: float (Default: None)
        batch_size: int (Default: 32)
        seed: int (Default: 666), random seed for shuffle or randomly sampling

    returns
    ---
        trainloader, valloader, testloader
    """

    train_data = pd.read_csv(os.path.join(path,'covid_train.csv'))
    train_data, val_data = split_train_val_test(train_data, val_rate, seed=seed)
    test_data = pd.read_csv(os.path.join(path,'covid_test.csv'))

    trainset = MyDataset(train_data.iloc[:,:-1], train_data.iloc[:, -1])
    valset = MyDataset(val_data.iloc[:,:-1], val_data.iloc[:, -1])
    testset = MyDataset(test_data.iloc[:, :])

    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=0, pin_memory=False)
    valloader = DataLoader(valset, batch_size, shuffle=True, num_workers=0, pin_memory=False)
    testloader = DataLoader(testset, batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return trainloader, valloader, testloader

def split_train_val_test(data, val_rate=0.1, test_rate=None, seed=666):
    r"""
    When test_rate is not None, split data into train, val and test sub set; or split data into train and val sub set.
    
    Args:
    ---
        data: pd.DataFrame
        val_rate: float (Default: 0.1)
        test_rate: float (Default: None)
        seed: int (Default: 666), random seed for shuffle or randomly sampling


    returns
    ---
        train_data, val_data, test_data(if test_rate not None)
    """

    train_rate = 1 - val_rate
    if test_rate is not None:
        train_rate -= test_rate
        train_data = data.sample(frac=train_rate, random_state=seed)
        val_data   = data.sample(frac=val_rate, random_state=seed)
        test_data  = data.sample(frac=test_rate, random_state=seed)

        return train_data, val_data, test_data
    else:
        train_data = data.sample(frac=train_rate, random_state=seed)
        val_data   = data.sample(frac=val_rate, random_state=seed)

        return train_data, val_data
    
class MyDataset(Dataset):
    def __init__(self, x, y=None):
        super().__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is not None:
            return self.x.iloc[index, :].values, self.y.iloc[index]
        else:
            return self.x.iloc[index, :].values