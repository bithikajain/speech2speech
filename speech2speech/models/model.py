#
# Copyright (C) 2020 Bithika Jain
#
from .encoder import Encoder
from .decoder import Decoder
from .vector_quantizer import VectorQuantizer

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, speaker_dic, speaker_embedding_dim, decay, device, verbose=False, debug=False):
        super(Model, self).__init__()
        self.verbose = verbose
        self.debug = debug
        self.speaker_dic = speaker_dic
        self.device = device

        self._encoder = Encoder(1, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                       commitment_cost, self.device)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens, speaker_dic, speaker_embedding_dim, self.device)

    def forward(self, x, speaker_id):
        if self.debug:
            print("Model::forward", x.size())
            sys.stdout.flush()
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        if self.debug:
            print("quantized ", quantized.size())
            sys.stdout.flush()
        x_recon = self._decoder(quantized, self.speaker_dic, speaker_id)

        input_features_size = x.size(2)
        if self.debug:
            print('input_features_size', input_features_size)
            sys.stdout.flush()
        output_features_size = x_recon.size(2)
        if self.debug:
            print('output_features_size', output_features_size)
            sys.stdout.flush()

        x_recon = x_recon.view(-1,1025, output_features_size)#1025 #257
        x_recon = x_recon[:, :, :-(output_features_size - input_features_size)]
        if self.debug:
            print('x_recon size', x_recon.size())
            sys.stdout.flush()

        return loss, x_recon, perplexity
