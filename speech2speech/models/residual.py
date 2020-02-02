import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens,
                 num_residual_hiddens, verbose=False, debug=False):

        super(Residual, self).__init__()
        self.verbose = verbose
        self.debug = debug

        self._block = nn.Sequential(

            nn.ReLU(True),

            nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),

            nn.ReLU(True),

            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1,
                      stride=1,
                      bias=False)
        )

    def forward(self, x):
        if self.debug:
            print('shape of x in Residual.forward', x.size())
            sys.stdout.flush()
        return x + self._block(x)
