#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i


class CNN(torch.nn.Module):

    def __init__(self, m_word, e_char, filter_num, kernel_size=5):
        super(CNN, self).__init__()
        self.in_channels = e_char
        self.out_channels = filter_num
        self.kernel_size = kernel_size
        self.conv1d = torch.nn.Conv1d(self.in_channels, self.out_channels, kernel_size)
        self.maxPool1d = torch.nn.MaxPool1d(m_word-kernel_size+1)
        self.relu = torch.nn.ReLU()

    def forward(self, x_reshape):
        """ caculate x_conv

        @param x_reshape:  shape:(N ,e_char ,m_word)

        @returns x_conv_out: shape:(N, filter_num)

        """
        x_conv = self.conv1d(x_reshape)  # (N, filter_num, m_word-k+1)
        x_relu = self.relu(x_conv)  # (N, filter_num, m_word-k+1)
        x_conv_out = self.maxPool1d(x_relu)
        x_conv_out = torch.squeeze(x_conv_out, dim=2)  # (N, filter_num)
        return x_conv_out

### END YOUR CODE

