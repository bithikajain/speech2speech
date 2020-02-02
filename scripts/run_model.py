###############################################################################
###############################################################################
#               Required Imports
#
import matplotlib.pyplot as plt
import math
import pandas as pd
from torchviz import make_dot, make_dot_from_trace
import random
from torchsummary import summary
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch
import umap
from six.moves import xrange
from scipy.signal import savgol_filter
import pathlib
import librosa.display
import librosa
import IPython.display as ipd  # To play sound in the notebook
import numpy as np
import json
import argparse
from speech2speech.data_preprocessing.load_data import *
from speech2speech.models.model import Model
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))


#import yaml


###############################################################################
###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
###############################################################################
#               Argument Parsing
#
parser = argparse.ArgumentParser(description='Train VQVAE for input data')
parser.add_argument('--verbose', action='store_true',
                    default=False, help='Verbose output (default: False)')
parser.add_argument('--debug', action='store_true',
                    default=False, help='Debugging (default: False)')
parser.add_argument('--base-dir', type=str,
                    default='/home/ubuntu/voice_conversion/', help='Input base directory (default: /home/ubuntu/voice_conversion/)')
parser.add_argument('--data-dir', type=str,
                    default='/home/ubuntu/voice_conversion/data/raw/VCTK-Corpus', help='Input data directory (default: /home/ubuntu/voice_conversion/data/raw/VCTK-Corpus)')
parser.add_argument('--spectrogram-dir', type=str, default='/home/ubuntu/voice_conversion/data/interim/spectogram_array_trim_30db',
                    help='Directory with spectrogram files (Default:/home/ubuntu/voice_conversion/data/interim/spectogram_array_trim_30db )')
parser.add_argument('--time_length', type=int, default=50,
                    help="Lenght of the time dimension for input spectrogram files (default: 50)")

parser.add_argument('--train-data-fraction', type=float,
                    default=0.8, help='train dataset split fraction (default: 0.8)')
parser.add_argument('--validation-data-fraction', type=float,
                    default=0.1, help='validation dataset split fraction (default: 0.1)')

parser.add_argument('--num-epochs', type=int,
                    default=20, help='number of epochs(default:20)')
parser.add_argument('--batch-size', type=int,
                    default=10, help='batch size (default:10)')
parser.add_argument('--num-training-updates', type=int,
                    default=15000)
parser.add_argument('--num-hiddens', type=int,
                    default=768)
parser.add_argument('--num-residual-hiddens', type=int,
                    default=32)
parser.add_argument('--num-residual-layers', type=int,
                    default=2)
parser.add_argument('--embedding-dim', type=int,
                    default=64)
parser.add_argument('--num-embeddings', type=int,
                    default=300)
parser.add_argument('--speaker-embedding-dim', type=int,
                    default=20)

parser.add_argument('--commitment-cost', type=float,
                    default=0.25, help='commitment cost (default: 0.25)')

parser.add_argument('--decay', type=float,
                    default=0, help='decay for VQVAE EM (default: 0)')

parser.add_argument('--learning-rate', type=float,
                    default=1e-4, help='learning rate (default: 1e-4)')
args = parser.parse_args()

###############################################################################
###############################################################################
# Set paths that you will use in this notebook
checkpoint_dir = os.path.join(args.base_dir, 'checkpoints')
plot_dir = os.path.join(args.base_dir, 'plots')
output_dir = os.path.join(args.base_dir, 'output')
os.system("mkdir -p {} {} {} {} {}".format(args.base_dir,
                                           args.data_dir,
                                           checkpoint_dir,
                                           plot_dir,
                                           output_dir))
if args.verbose:
    print("Made required directories...")
    sys.stdout.flush()

###############################################################################
###############################################################################
#                   Load dataset
#
files_np = list(glob.glob(os.path.join(args.spectrogram_dir, '*.*')))

tensordataset = spectrograms_to_torch_dataset(files_np, max_col)
speaker_dic = speaker_id_dic(files_np)
###############################################################################
###############################################################################
#                   Split data into train, test and validation
#

train_dataset, val_dataset, test_dataset = split_dataset(
    tensordataset, args.train_data_fraction, args.validation_data_fraction)

###############################################################################
###############################################################################
#               Data Loaders
#

training_loader, validation_loader = train_val_data_loaders(
    train_dataset, val_dataset, args.batch_size)

###############################################################################
###############################################################################
#           Create model
#
model = Model(args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens,
              args.num_embeddings, args.embedding_dim,
              args.commitment_cost, speaker_dic, args.speaker_embedding_dim, args.decay).to(device)
