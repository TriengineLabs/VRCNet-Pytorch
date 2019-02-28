#!/usr/bin/env python3

import sys
import pandas as pd
import librosa
import os
import ntpath
import h5py
import torch
import argparse
from tqdm import tqdm

from DeepUNet import DeepUNet
import train

parser = argparse.ArgumentParser(description='U-Net model for music source separation')
subparsers = parser.add_subparsers(dest='mode')

train_p = subparsers.add_parser('train')
train_p.add_argument('-d', '--data_path', required=True,
                     help='path to your preprocessed CSV data file')
train_p.add_argument('-e', '--epochs', default='5', help='Number of epochs to train', type=int)
gpu_group = train_p.add_mutually_exclusive_group()
gpu_group.add_argument('--cpu', action='store_true', help='train on CPU')
gpu_group.add_argument('--gpu', action='store_false', help='train on GPU')

preprocess_p = subparsers.add_parser('preprocess')
preprocess_p.add_argument('-d', '--data_path', required=True, help='path to your data directory')
preprocess_p.add_argument('-c', '--data_csv', required=True,
                          help='path to your CSV file linking paths of mixes and sources')
preprocess_p.add_argument('-o', '--out_dir', default='./numpy_data', help='Directory to save processed data')
preprocess_p.add_argument('-p', '--processed_csv_dir', default='./processed_dataset.csv',
                          help='Path to save processed CSV')

args = vars(parser.parse_args())


def prepare_dataset(data_path, dataset_csv, path_to_save='./numpy_data', processed_csv_path='./processed_dataset.csv'):
    print('Starting preparing dataset...')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    processed_csv = pd.DataFrame(columns=dataset_csv.columns)
    for item in tqdm(range(len(dataset_csv))):
        row_to_insert = []

        paths_of_mix = dataset_csv.iloc[item, :].values
        for i, p in enumerate(paths_of_mix):
            filename_with_ext = ntpath.basename(p)
            filename = os.path.splitext(filename_with_ext)[0]

            inp, sr = librosa.load(data_path + p)
            inp = librosa.resample(inp, sr, 8192)
            ft_inp = librosa.stft(inp, n_fft=1024, hop_length=768, window='hann', center=True)

            np_file_path = os.path.join(path_to_save, (filename + '.h5'))
            with h5py.File(np_file_path, 'w') as hf:
                hf.create_dataset('dataset', data=ft_inp)
            row_to_insert.append(np_file_path)

        processed_csv.loc[len(processed_csv)] = row_to_insert
    processed_csv.to_csv(processed_csv_path, index=False)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if args['mode'] == 'preprocess':
        # Read audio files once and store them with numpy extension for quicker processing during training
        # Make PREPARATION_NEEDED=True if dataset is new/changed, else set it False
        initial_data = pd.read_csv(args['data_csv'])
        prepare_dataset(args['data_path'], initial_data, args['out_dir'], args['processed_csv_dir'])
    elif args['mode'] == 'train':
        # Defining model
        model = DeepUNet(1, 1)
        # Start training
        train.train(model, args['data_path'], criterion=torch.nn.MSELoss(), gpu=args['gpu'], epochs=args['epochs'])