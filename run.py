#!/usr/bin/env python3

import sys
import torch
import argparse
import train
from torch.optim.lr_scheduler import StepLR
from model.VRCNet import VRCNet
from model.SCUNet import Generator
from model.VggUNet import VggUNet
from model.ResUNet import Generator as ResUNet
from model.VCNet import VCNet
from calculate_score import calculate_score
from torch.optim.lr_scheduler import StepLR
from pickle import UnpicklingError
from exceptions import StopTrainingException
from preprocess import prepare_dataset

torch.manual_seed(42)
torch.cuda.manual_seed(42)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='U-Net model for music source separation')
subparsers = parser.add_subparsers(dest='mode')

train_p = subparsers.add_parser('train')
train_p.add_argument('-d', '--data_path', required=True,
                     help='path to your preprocessed CSV data file')
train_p.add_argument('-v', '--valid_path', required=False,
                     help='path to your preprocessed validation CSV file', default=None)
train_p.add_argument('-e', '--epochs', default='5', help='Number of epochs to train', type=int)
train_p.add_argument('--lr', default=None, help='Learning Rate', type=float)
train_p.add_argument('--log_scale', default='False', help='Should the input be log scaled or not', type=str2bool)
train_p.add_argument('--batch_size', default=3, help='Batch Size', type=int)
train_p.add_argument('--model_weight_name', default='model_weights.pt', help='file name of Model Weights', type=str)
train_p.add_argument('--log_dir', default=None, help='Dir for logs', type=str)
train_p.add_argument('--log_name', default=None, help='Name for this experiment\'s log', type=str)
train_p.add_argument('--pretrained_model', default='', help='file name of PreTrained Weights to be loaded', type=str)
train_p.add_argument('--train_info_file', default=None, help='File to store training info', type=str)
train_p.add_argument('-j', '--workers', default=1, help='Number of workers', type=int)
train_p.add_argument('--model_name', default='SCUNet', help='File to store training info', type=str)

gpu_group = train_p.add_mutually_exclusive_group()
gpu_group.add_argument('--cpu', action='store_false', help='train on CPU')
gpu_group.add_argument('--gpu', action='store_true', help='train on GPU')

preprocess_p = subparsers.add_parser('preprocess')
preprocess_p.add_argument('-d', '--data_path', required=True, help='path to your data directory')
preprocess_p.add_argument('-s', '--data_subset', required=True,
                          help='path to your CSV file linking paths of mixes and sources')
preprocess_p.add_argument('-o', '--out_dir', default='./numpy_data', help='Directory to save processed data')
preprocess_p.add_argument('-p', '--processed_csv_dir', default='./processed_dataset.csv',
                          help='Path to save processed CSV')
preprocess_p.add_argument('-hl', '--hop_length', help='hop length value of stft', default=512)
preprocess_p.add_argument('-ws', '--n_fft', help='n_fft parameter  value of stft', default=2048)
preprocess_p.add_argument('-j', '--workers', default=1, help='Number of workers', type=int)
preprocess_p.add_argument('-sd', '--slice_duration', help='duration in seconds of slice to be cut before stft',
                          default=2)

test_p = subparsers.add_parser('test')
test_p.add_argument('--model_weight_name', required=True, help='File name of Model Weights', type=str)
test_p.add_argument('--data_path', required=True, help='Path to your data directory', type=str)
test_p.add_argument('-j', '--workers', default=1, help='Number of workers', type=int)
test_p.add_argument('--model_name', default='SCUNet', help='File to store training info', type=str)

args = vars(parser.parse_args())


def main():
    args = vars(parser.parse_args())
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if args['mode'] == 'preprocess':
        print('args data_path ', args['data_path'])
        print('args hop_length', args['hop_length'])
        # Read audio files once and store them with numpy extension for quicker processing during training
        # Make PREPARATION_NEEDED=True if dataset is new/changed, else set it False
        prepare_dataset(args['data_path'], args['data_subset'], args['out_dir'], args['processed_csv_dir'],
                        n_fft=args['n_fft'], hop_length=args['hop_length'], slice_duration=args['slice_duration'],
                        n_workers=args['workers'])
    elif args['mode'] == 'train':
        # Defining model
        if 'SCUNet' == args['model_name']:
            model = Generator(1)
        elif 'VggUNet' == args['model_name']:
            model = VggUNet()
        elif 'ResUNet' == args['model_name']:
            model = ResUNet()
        elif 'VRCNet' == args['model_name']:
            model = VRCNet()
        elif 'VCNet' == args['model_name']:
            model = VCNet()
        else:
            print('Sorry. That model currently is not implemented')
            return

        # If pre-trained weights are specified, load them:
        if args['pretrained_model']:
            try:
                model.load_state_dict(torch.load(args['pretrained_model']))
            except (UnpicklingError, FileNotFoundError) as e:
                print(e)
                print('The pretrained model path is not correct!')
                return
        # Start training
        train.train(model,
                    model_type=args['model_name'],
                    train_csv=args['data_path'],
                    validation_csv=args['valid_path'],
                    # scheduler=StepLR,
                    use_log_scale=args['log_scale'],
                    gpu=args['gpu'],
                    epochs=args['epochs'],
                    lr=args['lr'],
                    batch_size=args['batch_size'],
                    model_weight_name=args['model_weight_name'],
                    log_dir=args['log_dir'],
                    log_name=args['log_name'],
                    train_info_file=args['train_info_file'],
                    n_workers=args['workers'])
    elif args['mode'] == 'test':
        if 'SCUNet' == args['model_name']:
            model = Generator(1)
        elif 'VggUNet' == args['model_name']:
            model = VggUNet()
        elif 'ResUNet' == args['model_name']:
            model = ResUNet()
        else:
            print('Sorry. That model currently is not implemented')
            return
        calculate_score(model, model_weights_path=args['model_weight_name'], musdb_dir=args['data_path'],
                        n_workers=args['workers'])


if __name__ == "__main__":
    try:
        main()
    except StopTrainingException as e:
        print(e)
