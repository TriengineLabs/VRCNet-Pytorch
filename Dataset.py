from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import librosa
import torch
import os


class WaveDataset(Dataset):
    def __init__(self, dataframe, data_path='./Dataset', transforms=None):
        self.data_path = data_path
        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        paths = self.dataframe.iloc[item, :].values
        mel_specs = []
        for i, p in enumerate(paths):
            inp, sr = librosa.load(self.data_path + p)
            mlc, phase = librosa.magphase(librosa.stft(inp, n_fft=1024,hop_length=256,window='hann',center='True'))

            if i == 0 and self.transforms:
                mlc_tr = self.transforms(mlc)
                mel_specs.append(mlc_tr)
            mel_specs.append(mlc)

        return mel_specs
