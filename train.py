import torch
from icecream import ic
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import WaveDataset


def train(model,
          dataframe,
          epochs=5,
          gpu=True,
          optimizer=None,
          criterion=None,
          batch_size=1):

    device = 'cuda' if gpu else 'cpu'
    model.to(device)
    optimizer = optimizer if optimizer else optim.Adam(model.parameters())
    criterion = criterion if criterion else nn.L1Loss()
    dataset = WaveDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=1)
    for e in tqdm(range(epochs)):
        print('Starting Epoch', str(e) + '/' + str(epochs))
        epoch_loss = 0
        for lst in dataloader:
            optimizer.zero_grad()
            mask = model.forward(lst[0].float().to(device))
            mask = mask.squeeze(0)
            out = mask * lst[0].float().to(device)

            # TODO better handle unequal sizes for output and target
            loss = criterion(out, lst[1][:,:,:lst[0].shape[-1]].float().to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Epoch completed, Loss is: ', epoch_loss/batch_size)
