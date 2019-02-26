import torch
from icecream import ic
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import WaveDataset
import transforms


EARLY_STOPPING_EPOCHS = 20

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
    dataset = WaveDataset(dataframe, transforms=[transforms.HorizontalCrop(1024),
                                                 transforms.Normalize()])
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=3)
    
    unimproved_epochs = 0
    best_loss = 1000
    best_model_dict = None
    
    for e in range(epochs):
        print('Starting Epoch', str(e) + '/' + str(epochs))
        epoch_full_loss = 0
        for n_track, lst in enumerate(tqdm(dataloader)):
            #TODO change source hardcoding, handle unequal size of mix and source
            normalized_mix = lst[0].float().to(device)
            original_mix = lst[1].float().to(device)
            source1 = lst[2].float().to(device)

            # ic(normalized_mix.shape, original_mix.shape, source1.shape)

            optimizer.zero_grad()
            mask = model.forward(normalized_mix)
            mask = mask.squeeze(0)
            out = mask * original_mix

            loss = criterion(out, source1)
            loss.backward()
            optimizer.step()
            epoch_full_loss += loss.item()

        epoch_mean_loss = epoch_full_loss/len(dataloader)
        print('Epoch completed, Loss is: ', epoch_mean_loss)
        
        # Early Stopping
        if epoch_mean_loss > best_loss:
            unimproved_epochs += 1
            if unimproved_epochs > EARLY_STOPPING_EPOCHS:
                print('Early stopping happened')
                break
        else:
            best_model_dict = model.state_dict()
            unimproved_epochs = 0

    torch.save(best_model_dict, 'model_weights2.pt')

