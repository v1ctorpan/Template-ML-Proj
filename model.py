import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(88,40),
            nn.ReLU(),
            nn.Linear(40,1)
        )

    def forward(self,x):
        out = self.layer(x)

        return out.squeeze()

    