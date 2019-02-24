import torch
from icecream import ic
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from Dataset import WaveDataset





def normalize(vector):
    vector = torch.Tensor(vector)
    min_v = torch.min(vector)
    range_v = torch.max(vector) - min_v
    if range_v > 0:
        normalised = (vector - min_v) / range_v
    else:
        normalised = torch.zeros(vector.size())
    return normalised


def train(model,
          dataframe,
          epochs=15,
          gpu=True,
          optimizer=None,
          criterion=None,
          batch_size=1):

    device = 'cuda' if gpu else 'cpu'
    model.to(device)
    optimizer = optimizer if optimizer else optim.Adam(model.parameters())
    criterion = criterion if criterion else nn.L1Loss()
    dataset = WaveDataset(dataframe, transforms=normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=1)
    for e in range(epochs):
        print('Starting Epoch', str(e) + '/' + str(epochs))
        epoch_loss = 0
        for n_track, lst in enumerate(tqdm(dataloader)):
            #TODO change source hardcoding, handle unequal size of mix and source
            normalized_mix = lst[0].float().to(device)
            original_mix = lst[1].float().to(device)
            source1 = lst[2][:,:,:lst[0].shape[-1]].float().to(device)

            optimizer.zero_grad()
            mask = model.forward(normalized_mix)
            mask = mask.squeeze(0)
            out = mask * original_mix[:, :, :-1]

            loss = criterion(out, source1[:, :, :-1])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Epoch completed, Loss is: ', epoch_loss/batch_size)

    torch.save(model.state_dict(), 'model_weights.pt')

