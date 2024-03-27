import os
import sys

import torch
import torch.nn
import yaml
import time

from training import MyModel

# set file path to current work path
os.chdir(sys.path[0])

# load config file
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)

start = time.time()
print(f"Training starts! Now is {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start))}")

model = MyModel(config)
model.train()
model.test()

end = time.time()
print(f"Training is over! Now is {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end))}.")
print(f"Total duration is {end-start:.1f} seconds")