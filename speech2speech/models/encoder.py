from .residual_stack import ResidualStack

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
                 num_residual_hiddens, verbose=False, debug=False):
        super(Encoder, self).__init__()
        self.verbose = verbose
        self.debug = debug

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """

        self._conv_1 = nn.Conv1d(in_channels=1025,  # ??features_filters,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 padding=1)

        self._conv_2 = nn.Conv1d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 padding=1)
        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """

        self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=1,
                                 padding=2)
        """
        2 convolutional layers with length 3 and
        residual connections.
        """
        self._conv_4 = nn.Conv1d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 padding=1)

        self._conv_5 = nn.Conv1d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        if self.debug:
            print('shape of inputs in Encoder.forward', inputs.size())
            sys.stdout.flush()

        x_conv_1 = F.relu(self._conv_1(inputs))
        if self.debug:
            print('shape of x in Encoder.forward._conv_1', x_conv_1.size())
            sys.stdout.flush()

        x = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        if self.debug:
            print('shape of x in Encoder.forward.relu_2', x.size())
            sys.stdout.flush()

        x_conv_3 = F.relu(self._conv_3(x))
        if self.debug:
            print(
                'shape of x_conv_3 in Encoder.forward._conv_3',
                x_conv_3.size())
            sys.stdout.flush()
        x_conv_4 = F.relu(self._conv_4(x_conv_3)) + x_conv_3
        if self.debug:
            print(
                'shape of x_conv_4 in Encoder.forward._conv_4',
                x_conv_4.size())
            sys.stdout.flush()
        x_conv_5 = F.relu(self._conv_5(x_conv_4)) + x_conv_4
        if self.debug:
            print(
                'shape of x_conv_5 in Encoder.forward._conv_5',
                x_conv_5.size())
            sys.stdout.flush()
        x = self._residual_stack(x_conv_5) + x_conv_5
        if self.debug:
            print('shape of _residual_stack in Encoder.forward', x.size())
            sys.stdout.flush()
        return x
