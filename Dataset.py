from torch.utils.data import Dataset
import librosa


class WaveDataset(Dataset):
    def __init__(self, dataframe, data_path='Dataset/'):
        self.data_path = data_path
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        paths = self.dataframe.iloc[1, :].values
        mel_specs = []
        for p in paths:
            y, sr = librosa.load(self.data_path+p)
            mlc = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_specs.append(mlc)

        return mel_specs
