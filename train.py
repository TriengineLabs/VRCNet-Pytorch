import json
import os

import torch
from icecream import ic
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import WaveDataset
from exceptions import StopTrainingException
import transforms
import pandas as pd
from copy import deepcopy
from tensorboard_logger import configure, log_value

EARLY_STOPPING_EPOCHS = 100
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def saveInfoFile(train_info_file, details):
    a = []
    if not os.path.isfile(train_info_file):
        a.append(details)
        with open(train_info_file, mode='w') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open(train_info_file) as feedsjson:
            feeds = json.load(feedsjson)

        feeds.append(details)
        with open(train_info_file, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))

def train(model,
          csv_path,
          epochs=15,
          gpu=True,
          optimizer=None,
          criterion=None,
          scheduler=None,
          batch_size=1,
          model_weight_name='model_weights.pt',
          lr=None,
          log_dir=None,
          log_name=None,
          train_info_file=None):
    device = 'cuda' if gpu else 'cpu'
    model.to(device)

    if log_dir and log_name:
        configure(log_dir + '/' + log_name)
    elif (not log_dir and log_name) or (log_dir and not log_name):
        raise ValueError('Either both log_value and log_name or none of them shall be provided')

    optimizer = optimizer if optimizer else optim.Adam(model.parameters())
    if lr:
        for g in optimizer.param_groups:
            g['lr'] = lr

    details = {'CSV_path': csv_path,
               'epochs': epochs,
               'gpu': gpu,
               'optimizer': str(optimizer),
               'criterion': str(criterion),
               'lr': lr,
               'batch_size': batch_size,
               'model_weight_name': model_weight_name,
               'log_dir': log_dir,
               'log_name': log_name}

    criterion = criterion if criterion else nn.L1Loss()
    scheduler = scheduler(optimizer, step_size=100, gamma=0.999) if scheduler else None

    procesed_data = pd.read_csv(csv_path)
    dataset = WaveDataset(procesed_data, transforms=[transforms.HorizontalCrop(449),
                                                 transforms.Normalize()])
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=3)

    unimproved_epochs = 0
    best_loss = 1000
    best_model_dict = None

    for e in range(epochs):
        try:
            print('Starting Epoch', str(e) + '/' + str(epochs))
            epoch_full_loss = 0
            for n_track, lst in enumerate(tqdm(dataloader)):
                # TODO change source hardcoding, handle unequal size of mix and source
                normalized_mix = lst[0].float().to(device)
                original_mix = lst[1].float().to(device)
                source1 = lst[2].float().to(device)

                optimizer.zero_grad()
                mask = model.forward(normalized_mix)
                mask = mask.squeeze(0).squeeze(1)
                out = mask * original_mix

                loss = criterion(out, source1)
                loss.backward()
                optimizer.step()
                epoch_full_loss += loss.item()

                if scheduler:
                    scheduler.step()

            epoch_mean_loss = epoch_full_loss / len(dataloader)
            if log_dir and log_name:
                log_value('Training Epoch Loss', epoch_mean_loss)
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
                details['min_loss'] = best_loss
                details['stopped_on'] = e
                saveInfoFile(train_info_file, details)
            raise StopTrainingException(e)

    details['min_loss'] = best_loss
    saveInfoFile(train_info_file, details)

    torch.save(best_model_dict, model_weight_name)
