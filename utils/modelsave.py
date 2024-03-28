import os
import yaml

import matplotlib.pyplot as plt
import numpy as np
import torch

class ModelSave:
    def __init__(self, config, checkpoints, epochs, path, tag, patience=100, verbose=False, delta=0.0):
        self.config = config
        self.checkpoints = np.linspace(0,epochs,checkpoints+1)[1:]
        self.checkpoints[-1] -= 1
        self.path = path
        self.tag_idx = self.gettagidx(tag)
        if not os.path.exists(os.path.join(path,self.tag_idx)):
            os.mkdir(os.path.join(path,self.tag_idx))
        with open(os.path.join(f'{path}/{self.tag_idx}','hparam.yaml'), encoding='utf-8',mode='w') as f:
            yaml.dump(self.config,f,allow_unicode=True)

        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc, epoch, model, optimizer):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, epoch, model, optimizer, True)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(score, epoch, model, optimizer, True)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            self.save_checkpoint(score, 'last', model, optimizer)
        if epoch in self.checkpoints:
            self.save_checkpoint(score, epoch, model, optimizer)

        return self.early_stop

    def save_checkpoint(self, val_acc, epoch, model, optimizer, tag=False):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print(f'Validation acc increased ({self.val_acc:.6f} --> {val_acc:.6f}).  Saving model ...')
        
        checkpoint = {"model":model,
                      "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch}
        
        if not tag: # 说明是到检查点要保存
            if epoch==self.checkpoints[-1]:
                path_checkpoint = f'{self.path}/{self.tag_idx}/checkpoint_last_epoch.pkl'
            else:
                path_checkpoint = f'{self.path}/{self.tag_idx}/checkpoint_{epoch}_epoch.pkl'
            torch.save(checkpoint, path_checkpoint)
        else: # 说明是best保存
            path_checkpoint = f'{self.path}/{self.tag_idx}/checkpoint_best_epoch.pkl'
            torch.save(checkpoint, path_checkpoint)

    def gettagidx(self, tag):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        log_files = os.listdir(self.path)
        idx = []
        for i in log_files:
            s = i.split("_")
            if s[0] == tag:
                idx.append(int(s[-1]))
        tag_idx = f'{tag}_0' if idx == [] else f'{tag}_{max(idx)+1}'
        return tag_idx