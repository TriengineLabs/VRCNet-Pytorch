import train
import pandas as pd
from DeepUNet import DeepUNet

data = pd.read_csv('dataset_info.csv') 
model = DeepUNet(1, 1)
train.train(model, data, gpu=True, epochs = 20)
