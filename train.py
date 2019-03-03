import torch
from icecream import ic
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import WaveDataset
import transforms
import pandas as pd
from copy import deepcopy

EARLY_STOPPING_EPOCHS = 100
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def train(model,
          csv_path,
          epochs=15,
          gpu=True,
          optimizer=None,
          criterion=None,
          scheduler=None,
          batch_size=1,
          model_weight_name='model_weights.pt',
          lr=None):
    device = 'cuda' if gpu else 'cpu'
    model.to(device)
    optimizer = optimizer if optimizer else optim.Adam(model.parameters())
    if lr:
        for g in optimizer.param_groups:
            g['lr'] = lr

    criterion = criterion if criterion else nn.L1Loss()
    scheduler = scheduler(optimizer, step_size=50, gamma=0.97) if scheduler else None

    procesed_data = pd.read_csv(csv_path)
    dataset = WaveDataset(procesed_data, transforms=[transforms.HorizontalCrop(449),
                                                 transforms.Normalize()])
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=3)

    unimproved_epochs = 0
    best_loss = 1000
    best_model_dict = None

    try:
        for e in range(epochs):
            print('Starting Epoch', str(e) + '/' + str(epochs))
            epoch_full_loss = 0
            for n_track, lst in enumerate(tqdm(dataloader)):
                # TODO change source hardcoding, handle unequal size of mix and source
                normalized_mix = lst[0].float().to(device)
                original_mix = lst[1].float().to(device)
                source1 = lst[2].float().to(device)

                optimizer.zero_grad()
                mask = model.forward(normalized_mix)
                mask = mask.squeeze(0)
                out = mask * original_mix

                loss = criterion(out, source1)
                loss.backward()
                optimizer.step()
                epoch_full_loss += loss.item()

                if scheduler:
                    scheduler.step()

            epoch_mean_loss = epoch_full_loss / len(dataloader)
            print('Epoch completed, Loss is: ', epoch_mean_loss)

            # Early Stopping
            if epoch_mean_loss > best_loss:
                unimproved_epochs += 1
                if unimproved_epochs > EARLY_STOPPING_EPOCHS:
                    print('Early stopping happened')
                    break
            else:
                best_model_dict = deepcopy(model.state_dict())
                best_loss = epoch_mean_loss
                unimproved_epochs = 0
    except KeyboardInterrupt:
        if best_model_dict:
            print('Saving the model!!')
            torch.save(best_model_dict, ('interrupted_' + model_weight_name))
        raise (KeyboardInterrupt)
    torch.save(best_model_dict, model_weight_name)
