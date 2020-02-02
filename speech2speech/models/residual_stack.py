#
# Copyright (C) 2020 Bithika Jain
#
from .residual import Residual

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
                 num_residual_hiddens, verbose=False, debug=False):
        super(ResidualStack, self).__init__()
        self.verbose = verbose
        self.debug = debug

        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        if self.debug:
            print('shape of x in ResidualStack.forward', x.size())
            sys.stdout.flush()
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
            if self.debug:
                print(
                    f'Iteration {i} shape of x in ResidualStack.forward {x.size()} ',)
                sys.stdout.flush()
        return F.relu(x)
