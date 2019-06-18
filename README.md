# SCUNet-Pytorch

Implementation of the paper "Singing Voice Separation using U-Net based architectures".

## Dataset

This repository uses **musdb** dataset for training and evaluation. The dataset is free, but requires access. For downloading and for further usage of the dataset check [this](https://zenodo.org/record/1117372#.XQlP9bpfg3E) webpage.

## Preparation

The repository is constomized on **musdb** dataset and requires it to be downloaded and saved in a convenient folder. It is also needed to do preprocessing, as reading the audio files in each iteration may require huge amount of time. During the preprocess step, audio files are cutted into 2 second parts (by default, but can be changed), resampled if needed, passed through short-time Fourier transform and saved as an .h5 file. For preparing dataset, it is needed to call

`python3 run.py preprocess  -d *DatasetPath* -s *train/test* -o *ProcessedDatasetPath*`

`-d [REQUIRED] parameter specifies the extracted musdb dataset path with train and test folders in it`

`-s [REQUIRED] parameter specifies the subset of the dataset to be processed, i.e train or test`

`-o [REQUIRED] parameter sepcifies the folder, where processed .h5 files will be saved`

`-p -> csv file path that should be generated that will point to the processed files. Default is ./processed_dataset.csv`

There are also some optional parameters that change default values of short-time Fourier transform, as well as slice duration during preparation.

## Training

For training, the csv file generated during Preparation step is mandatory.

`./run.py train -d *CSVPath* -e *NumberOfEpochs* --model_name=*ModelName* --model_weight_name=*SaveModelAs* --batch_size=*SizeOfTheBatch* --train_info_file=*LogPath*`
