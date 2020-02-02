#
# Copyright (C) 2020 Bithika Jain
#


import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def std_mean_dataloader(data_loader):
    mean = 0.
    meansq = 0.
    for data in data_loader:
        mean = data[0].mean()
        meansq = (data[0]**2).mean()
    std = torch.sqrt(meansq - mean**2)

    return std, mean 


def train_model(model, optimizer, num_epochs, training_loader):


    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    loss = []

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
           
            if (i+1) % 10 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], recon_error: {:.4f}, perplexity: {:.4f}" 
                   .format(epoch+1, num_epochs, i+1, len(training_loader), recon_error.item(), perplexity.item()))
          
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

    return train_res_recon_error, train_res_perplexity
