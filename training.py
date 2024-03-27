import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import Model
from datawork import dataprocess

class MyModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'])
        MyModel.get_gpu_property(self.device)

        self.model = Model().to(self.device)
        self.trainloader, self.valloader, self.testloader = dataprocess(
            path=config['data']['path'],
            val_rate=config['data']['val_rate'],
            batch_size=config['data']['batch_size'],
            seed=config['others']['seed']
        )
        
    def train(self):   
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['training']['lr'])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,1e-3,1e-5)
        criterion = nn.MSELoss()
        
        epochs = self.config['training']['epochs']
        trainbar = tqdm(range(epochs))

        for epoch in trainbar:
            trainbar.set_description(f'Epoch {epoch+1}/{epochs}')

            self.model.train()
            loss_record = []
            for data in tqdm(self.trainloader, desc='Training', leave=False):
                x, y = data
                x, y = x.float().to(self.device), y.float().to(self.device)

                optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                loss_record.append(loss.detach().item())
            train_loss = np.mean(loss_record)

            self.model.eval()
            loss_record = []
            for data in tqdm(self.valloader, desc='Validating', leave=False):
                x, y = data
                x, y = x.float().to(self.device), y.float().to(self.device)

                with torch.no_grad():
                    pred = self.model(x)
                    loss = criterion(pred, y)
                
                    loss_record.append(loss.detach().item())
            val_loss = np.mean(loss_record)

            trainbar.set_postfix({'train loss': train_loss, 'val loss': val_loss})

    def test(self):
        criterion = nn.MSELoss()

        self.model.eval()
        pred_record = []
        for data in tqdm(self.testloader, desc='Testing', leave=False):
            x = data
            x = x.float().to(self.device)

            with torch.no_grad():
                pred = self.model(x)
                pred_record += pred.detach().tolist()
                # if y is not None:
                #     loss = criterion(pred, y)
                #     loss_record.append(loss.detach().item())
        # if not loss_record:
        #     test_loss = np.mean(loss_record)
        #     print(f'Testing Finished! Test loss: {test_loss}')
        
        MyModel.savetest(pred_record, self.config['others']['save_path'])
    
    @staticmethod
    def savetest(pred, filename):
        df = pd.DataFrame(pred, columns=['tested_positive'])
        df.to_csv(filename, index_label='id')

    @staticmethod
    def get_gpu_property(device):
        properties = torch.cuda.get_device_properties(device)

        print("{:*^40}".format("GPU Info"))
        print("GPU Device: ", properties.name)
        print("Total memory: ", properties.total_memory / (1024**2), "MB")  # 转换为 MB 单位
        print("CUDA capability: ", properties.major, ".", properties.minor)
        print("{:*^40}".format("*"))



