import train
import pandas as pd
from DeepUNet import DeepUNet
import librosa
import os
import ntpath
import h5py
import torch.nn.functional as F
import torch

PREPARATION_NEEDED = False

INITIAL_DATASET_CSV_PATH = 'dataset_info.csv'
INITIAL_DATASET_PATH = './Dataset'
PROCESSED_DATASET_CSV_PATH = 'processed_dataset.csv'

def prepare_dataset(data_path, dataset_csv, path_to_save='./numpy_data', processed_csv_path='./processed_dataset.csv'):
    print('starting preparing dataset')
    processed_csv = pd.DataFrame(columns=dataset_csv.columns)
    for item in range(len(dataset_csv)):
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
    # Read audio files once and store them with numpy extension for quicker processing during training
    # Make PREPARATION_NEEDED=True if dataset is new/changed, else set it False
    if PREPARATION_NEEDED:
        initial_data = pd.read_csv(INITIAL_DATASET_CSV_PATH)
        prepare_dataset(INITIAL_DATASET_PATH, initial_data)

    # Defining model
    model = DeepUNet(1, 1)
    # Start training
    train.train(model, PROCESSED_DATASET_CSV_PATH, criterion=torch.nn.MSELoss(), gpu=True, epochs =200)
