import os
from tqdm import tqdm
import musdb
import h5py
import librosa
import pandas as pd

def prepare_dataset(data_path, subset=None, path_to_save='./numpy_data', processed_csv_path='./processed_dataset.csv'):
    mus = musdb.DB(root_dir='musdb')
    music_list = mus.load_mus_tracks(subsets='train')
    print('Starting preparing dataset...')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    processed_csv = pd.DataFrame(columns=['mix'] + list(music_list[0].targets.keys()))
    for item in tqdm(range(len(music_list))):
        row_to_insert = []

        # paths_of_mix = dataset_csv.iloc[item, :].values
        audio = music_list[item]
        for i, p in enumerate(processed_csv.columns):
            if p == 'mix':
                inp = librosa.to_mono(audio.audio.transpose())
            else:
                inp = librosa.to_mono(audio.targets[p].audio.transpose())

            sr = audio.rate
            inp = librosa.resample(inp, sr, 8192)
            ft_inp = librosa.stft(inp, n_fft=1024, hop_length=768, window='hann', center=True)

            filename = audio.name + '.' + p

            np_file_path = os.path.join(path_to_save, (filename + '.h5'))
            with h5py.File(np_file_path, 'w') as hf:
                hf.create_dataset('dataset', data=ft_inp)
            row_to_insert.append(np_file_path)

        processed_csv.loc[len(processed_csv)] = row_to_insert
    processed_csv.to_csv(processed_csv_path, index=False)
