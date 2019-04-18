import json
import os
import datetime

from tqdm import tqdm
from icecream import ic
import pandas as pd
from copy import deepcopy
# from tensorboard_logger import configure, log_value

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from Dataset import WaveDataset
from exceptions import StopTrainingException
import transforms

EARLY_STOPPING_EPOCHS = 100
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def saveInfoFile(train_info_file, details):
    a = []
    details['end_time'] = str(datetime.datetime.now())
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
          model_type,
          train_csv,
          validation_csv=None,
          epochs=15,
          gpu=True,
          optimizer=None,
          criterion=None,
          scheduler=None,
          use_log_scale=False,
          batch_size=1,
          model_weight_name='model_weights.pt',
          lr=None,
          log_dir=None,
          log_name=None,
          train_info_file=None,
          n_workers=1):
    device = torch.device('cuda') if gpu else torch.device('cpu')
    model.to(device)

    if log_dir and log_name:
        configure(log_dir + '/' + log_name)
    elif (not log_dir and log_name) or (log_dir and not log_name):
        raise ValueError('Either both log_value and log_name or none of them shall be provided')

    optimizer = optimizer if optimizer else optim.Adam(model.parameters())
    if lr:
        for g in optimizer.param_groups:
            g['lr'] = lr

    criterion = criterion if criterion else nn.L1Loss()
    scheduler = scheduler(optimizer, step_size=100, gamma=0.999) if scheduler else None
    train_info_file = train_info_file if train_info_file else model_weight_name + '.log'

    details = {'train_csv_path': train_csv,
               'validation_csv_path': train_csv,
               'start_time': str(datetime.datetime.now()),
               'epochs': epochs,
               'gpu': gpu,
               'optimizer': str(optimizer),
               'criterion': str(criterion),
               'scheduler': str(scheduler),
               'lr': lr,
               'batch_size': batch_size,
               'model_weight_name': model_weight_name,
               'log_dir': log_dir,
               'log_name': log_name}

    train_data = pd.read_csv(train_csv)

    transforms_to_do = [transforms.Normalize()]
    if model_type == 'VSegm':
        transforms_to_do.append(transforms.Resize(224, 224))

    dataset = WaveDataset(train_data,
                          # transforms=[transforms.HorizontalCrop(128),
                          # transforms.Normalize()],
                          # use_log_scale = use_log_scale)
                          transforms=transforms_to_do,
                          use_log_scale=False)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=n_workers)
    if validation_csv:
        valid_data = pd.read_csv(validation_csv)
        valid_dataset = WaveDataset(valid_data,
                                    transforms=transforms_to_do,
                                    use_log_scale=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=n_workers)
    unimproved_epochs = 0
    best_loss = 1000
    best_model_dict = None
    epoch_mean_loss = None
    valid_mean_loss = None

    for e in range(epochs):
        model.train()
        try:
            print('Starting Epoch', str(e) + '/' + str(epochs))
            epoch_full_loss = 0
            for n_track, lst in enumerate(tqdm(dataloader)):
                # TODO change source hardcoding, handle unequal size of mix and source
                normalized_mix = lst[0].float().to(device)
                original_mix = lst[1].float().to(device)
                source1 = lst[2].float().to(device)

                normalized_mix = normalized_mix.unsqueeze(1)
                optimizer.zero_grad()

                x = normalized_mix
                # if 'VSegm' == model_type:
                #     x = torch.cat((x, x, x), 1)

                mask = model.forward(x)
                mask = mask.squeeze(1)
                # ic(mask.shape, original_mix.shape, normalized_mix.shape)
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

            if validation_csv:
                valid_full_loss = 0
                model.eval()
                for n_track, lst in enumerate(dataloader):
                    with torch.no_grad:
                        normalized_mix = lst[0].float().to(device)
                        original_mix = lst[1].float().to(device)
                        source1 = lst[2].float().to(device)
                        normalized_mix = normalized_mix.unsqueeze(1)
                        mask = model.forward(normalized_mix.squeeze(1))
                        mask = mask.squeeze(1)
                        out = mask * original_mix
                        loss = criterion(out, source1)
                        valid_full_loss += loss.item()

                valid_mean_loss = epoch_full_loss / len(dataloader)

                print('Epoch completed, Training Loss: ', epoch_mean_loss, '\tValidation loss: ', valid_mean_loss)
            else:
                print('Epoch completed, Training Loss: ', epoch_mean_loss)

            # Early Stopping
            eval_loss = valid_mean_loss if validation_csv else epoch_mean_loss
            if eval_loss > best_loss:
                unimproved_epochs += 1
                if unimproved_epochs > EARLY_STOPPING_EPOCHS:
                    print('Early stopping happened')
                    break
            else:
                best_model_dict = deepcopy(model.state_dict())
                print('Saving the model!!')
                torch.save(best_model_dict, ('incomplete_' + model_weight_name))
                best_loss = eval_loss
                unimproved_epochs = 0
        except KeyboardInterrupt:
            if best_model_dict():
                print('Saving the model!!')
                torch.save(best_model_dict, ('interrupted_' + model_weight_name))
                details['eval_loss'] = best_loss
                details['train_loss'] = epoch_mean_loss
                details['valid_loss'] = valid_mean_loss

                details['stopped_on'] = e
                saveInfoFile(train_info_file, details)
            raise StopTrainingException(e)

    details['eval_loss'] = best_loss
    details['train_loss'] = epoch_mean_loss
    details['valid_loss'] = valid_mean_loss

    saveInfoFile(train_info_file, details)

    torch.save(best_model_dict, model_weight_name)
