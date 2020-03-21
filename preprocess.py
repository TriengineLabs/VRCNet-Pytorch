import os
import torchaudio
import torch
from tqdm import tqdm
import musdb
import h5py
import pandas as pd
from transforms import HorizontalCrop
import parmap
from icecream import ic


def prepare_dataset(data_path, subset=None,
                    path_to_save='./numpy_data',
                    processed_csv_path='./processed_dataset.csv',
                    resample_rate=None,
                    n_fft=2048,
                    hop_length=512,
                    slice_duration=2,
                    n_workers=1):
    print('hop_length = ', hop_length)
    mus = musdb.DB(root=data_path)
    music_list = mus.load_mus_tracks(subsets=subset)
    print('Starting preparing dataset...')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    processed_csv = pd.DataFrame(columns=['mix'] + list(music_list[0].targets.keys()))
    # p = multiprocessing.Pool(6)
    rows = parmap.map(process_audio, music_list, processed_csv, pm_pbar=True,
                      pm_processes=n_workers, path_to_save=path_to_save, n_fft=n_fft,
                      resample_rate=resample_rate, hop_length=hop_length, slice_duration=slice_duration)
    for r in rows:
        for n in r:
            processed_csv.loc[len(processed_csv)] = n

    processed_csv.to_csv(processed_csv_path, index=False)


def process_audio(audio, processed_csv,
                  path_to_save='./numpy_data',
                  resample_rate=None,
                  n_fft=2048,
                  hop_length=512,
                  slice_duration=2):
    rows = []

    for i, p in enumerate(processed_csv.columns):
        if p == 'mix':
            inp = torch.mean(torch.tensor(audio.audio.transpose()), dim=0)

        else:
            try:
                inp = torch.mean(torch.tensor(audio.targets[p].audio.transpose()), dim=0)
            except ValueError:
                print(f"Error: could not process target {p} from audio {audio.name}. Skipping...")
        sr = audio.rate
        if len(inp) < slice_duration:
            return
        for tr in range(len(inp) // (slice_duration * sr)):
            inp_slice = inp[tr * sr * slice_duration:(tr + 1) * slice_duration * sr]
            if resample_rate:
                inp_slice = torchaudio.transforms.Resample(sr, resample_rate).forward(inp)
            ft_inp = torch.stft(inp_slice, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft), center=True)
            # print(ft_inp_slice.shape)
            filename = audio.name + '.' + p + '_' + str(tr)

            np_file_path = os.path.join(path_to_save, (filename + '.h5'))
            with h5py.File(np_file_path, 'w') as hf:
                hf.create_dataset('dataset', data=ft_inp)
            if len(rows) > tr:
                rows[tr].append(np_file_path)
            else:
                rows.append([np_file_path])

    return rows
