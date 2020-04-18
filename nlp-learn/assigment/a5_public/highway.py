#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h


class Highway(torch.nn.Module):

    def __init__(self, D_in):
        super(Highway, self).__init__()
        self.linearProj = torch.nn.Linear(D_in, D_in, bias=True)
        self.linearGate = torch.nn.Linear(D_in, D_in, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x_conv_out):
        x_proj = self.relu(self.linearProj(x_conv_out))
        x_gate = torch.sigmoid(self.linearGate(x_conv_out))

        x_highway = x_gate * x_proj + (1-x_gate) * x_conv_out

        return x_highway


### END YOUR CODE

