#
# Copyright (C) 2020 Bithika Jain
#
import os
import math
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def spectrograms_to_torch(file, max_col):
    """
    Generates torch tensors for the spectrogram array and return
    them as a dataset
    """
    load_x = np.load(file)
    x = np.zeros((load_x.shape[0], 100), dtype=np.float32)
    c = min(100, load_x.shape[1])
    x[:load_x.shape[0], :c] = load_x[:, :c]
    x = torch.tensor(x, device='cpu').float()
    trial_y = file.split('/p')[1][:3]
    trial_y = torch.tensor(int(trial_y), device='cpu').float()
    return x, trail_y

def spectrograms_to_torch_dataset(files_np, max_col):
    """
    Generates torch tensors for the spectrogram array and return
    them as a dataset
    """
    X_list = []
    Y_list = []
    for file in files_np:
        load_x = np.load(file)
        x = np.zeros((load_x.shape[0], max_col), dtype=np.float32)
        c = min(max_col, load_x.shape[1])
        x[:load_x.shape[0], :c] = load_x[:, :c]

        x = torch.tensor(x, device='cpu').float()
        trial_y = file.split('trim_spec_p')[1][:3]#split('/p')[1][:3]
        trial_y = torch.tensor(int(trial_y), device='cpu').float()
        X_list.append(x)
        Y_list.append(trial_y)

    x_Tensor = torch.stack(X_list)
    y_Tensor = torch.stack(Y_list)
    tensordataset = torch.utils.data.TensorDataset(x_Tensor, y_Tensor)
    return tensordataset


def speaker_id_dic(files_np):
    """
    Generates speaker dictionary to be used for speaker embedding
    """

    Y_np_list = []
    for file in files_np:
        trial_y = file.split('trim_spec_p')[1][:3]
        Y_np_list.append(trial_y)
    Y_np_list = list(set(Y_np_list))
    speaker_dic = {Y_np_list[i]: i for i in range(0, len(Y_np_list))}
    return speaker_dic


def split_dataset(full_tensordataset, train_data_fraction=0.8,
                  validation_data_fraction=0.1):
    train_val_data_fraction = train_data_fraction + validation_data_fraction
    assert(train_val_data_fraction < 1.0, 'Check data split fractions')
    train_val_size = int(train_val_data_fraction * len(full_tensordataset))
    test_size = len(full_tensordataset) - train_val_size
    train__val_dataset, test_dataset = torch.utils.data.random_split(
        full_tensordataset, [train_val_size, test_size])
    train_size = int(
        (train_data_fraction /
         train_val_data_fraction) *
        len(train__val_dataset))
    val_size = len(train__val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train__val_dataset, [train_size, val_size])

    assert(len(test_dataset) + len(train_dataset) + len(val_dataset)
           == len(full_tensordataset), 'Data not split correctly ')
    return train_dataset, val_dataset, test_dataset


def train_val_data_loaders(train_dataset, val_dataset, batch_size):
    training_loader = DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)
    validation_loader = DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    return training_loader, validation_loader
