#
# Copyright (C) 2020 Bithika Jain
#
import os, sys
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

def std_mean_dataloader(data_loader):
    mean = 0.
    meansq = 0.
    for data in data_loader:
        mean = data[0].mean()
        meansq = (data[0]**2).mean()
    std = torch.sqrt(meansq - mean**2)

    return std, mean


def train_model(model, optimizer, num_epochs, training_loader, device, checkpoint_dir):
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    loss = []
    epoch_loss = 0
    for epoch in range(num_epochs):
        std, _ = std_mean_dataloader(training_loader)
        for i, (x, _) in enumerate(training_loader):
            batch_X = x.to(device)
            # forward pass
            vq_loss, data_recon, perplexity = model(batch_X)
            recon_error = F.mse_loss(data_recon, x.cuda()) / std
            loss = recon_error + vq_loss
            # Backprop and optimize
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (i+1) % 5 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], recon_error: {:.4f}, perplexity: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, len(training_loader), recon_error.item(), perplexity.item()))
        torch.save(model, os.path.join(checkpoint_dir, 'checkpoint_model_{}.pth'.format(epoch+1)))
        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
		 'model_parameters': model.named_parameters() }
        torch.save(state, os.path.join(checkpoint_dir,'checkpoint_state_dict_{}.pth'.format(epoch+1)))

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

    return train_res_recon_error, train_res_perplexity

def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model_params_dict = checkpoint['model_params_dict']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model_params_dict, model, optimizer, start_epoch


def eval_model(model, training_loader, device, checkpoint_dir):
    model.eval() # enter evaluation mode
    val_res_recon_error = []
    val_res_perplexity = []
    epoch_loss = 0
    with torch.no_grad():
        for epoch in range(num_epochs):
            std, _ = std_mean_dataloader(validation_loader)
            for i, (x, _) in enumerate(validation_loader):
                batch_X = x.to(device)
                # forward pass
                vq_loss, data_recon, perplexity = model(batch_X)
                recon_error = F.mse_loss(data_recon, x.cuda()) / std
                loss = recon_error + vq_loss
                epoch_loss += loss.item()

                if (i+1) % 10 == 0:
                    print ("Epoch[{}/{}], Step [{}/{}], recon_error: {:.4f}, perplexity: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, len(training_loader), recon_error.item(), perplexity.item()))


            val_res_recon_error.append(recon_error.item())
            val_res_perplexity.append(perplexity.item())
