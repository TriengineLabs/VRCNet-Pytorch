from torch.utils.data import Dataset
import librosa


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
            mlc, phase = librosa.magphase(librosa.stft(inp, n_fft=1024, hop_length=256, window='hann', center=True))
            mel_specs.append(mlc)

        # TODO add pipeline
        if self.transforms:
            for tr in self.transforms:
                mel_specs = tr(mel_specs)

        return mel_specs
