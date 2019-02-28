from torch.utils.data import Dataset
import librosa
import h5py


class WaveDataset(Dataset):
    def __init__(self, dataframe, transforms=None):
        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        paths = self.dataframe.iloc[item, :].values
        mel_specs = []
        for i, p in enumerate(paths):
            # Read saved numpy arrays that correspond to the initial music
            with h5py.File(p, 'r') as hf:
                data = hf['dataset'][:]
            mlc, phase = librosa.magphase(data)
            #TODO find better way to handle even shape
            if mlc.shape[1]%2 == 0:
                mlc = mlc[:, :-1]
            mel_specs.append(mlc)

        # TODO add pipeline
        if self.transforms:
            for tr in self.transforms:
                mel_specs = tr(mel_specs)

        return mel_specs
