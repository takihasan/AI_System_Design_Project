#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch import nn


# abstract quantizaiton class
class QLayer(nn.Module):
    def __init__(self, bits, signed=True):
        super().__init__()
        self.bits = bits

        val = 2 ** (bits - 1)

        if signed:
            self.n = -val
            self.p = val - 1
        else:
            self.n = 0
            self.p = 2 * val - 1

        self.s = None
        self.loss = 0.0

    def initialize_s(self, x):
        # initialize step
        val = 2*torch.mean(x.view(-1)) / torch.sqrt(self.p)
        self.s = nn.Parameter(torch.tensor([val]))

    def bins(self, x):
        return torch.clip(x/self.s, self.n, self.p).round()  # equation 1

    def clip(self, x):
        return self.bins(x) * self.s  # equation 2

    def forward(self, x):
        if self.s is None:
            self.initialize_s(x)
        x = self.clip(x)
        x = self.net(x)
        self.loss += self.br_loss(x)
        return x

    def br_loss(self, x):
        # x is tensor of values -> full precision
        x = x.view(-1)  # flatten
        # make bins -> integers [0, 1, 2, 1, 0, ...]
        bins = self.bins(x)
        bin_center = self.n + self.s / 2
        loss = 0.0
        # slow because it's a for loop
        for i in torch.arange(self.n, self.p):
            # bins == i -> all the positions where x is in bin <i>
            # take x values in bin <i>
            bin_content = x[bins == i]
            mse_loss = ((bin_content - bin_center) ** 2).mean()  # mean squared error
            var_loss = torch.var(bin_content) if len(bin_content) > 1 else 0.0
            loss += mse_loss + var_loss  # sum for all the bins
            bin_center += self.s  # next bin center
        return loss


class QLinear(QLayer):
    def __init__(self, in_size, out_size, bits=4):
        super().__init__(bits)
        self.net = nn.Linear(in_size, out_size)


class QConv2d(QLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bits=4,
                 ):
        super().__init__(bits)
        self.net = nn.Conv2d(in_channels, out_channels,
                             kernel_size, stride, padding)


class Model(nn.Module):
    def __init__(self, out_size=10, bits=4):
        super().__init__()
        net = []
        net.append(QConv2d(1, 6, 5, bits=bits))  # index 0
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(2))
        net.append(QConv2d(6, 16, 5, bits=bits))  # index 3
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(2))
        net.append(nn.Flatten())
        net.append(QLinear(256, 120))  # index 7
        net.append(nn.ReLU())
        net.append(QLinear(120, 84))  # index 9
        net.append(nn.ReLU())
        net.append(QLinear(84, out_size))  # index 10
        self.net = nn.Sequential(*net)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

    def loss(self, x):
        loss = 0
        for i in [0, 3, 7, 9, 10]:
            loss += self.net[i].loss
            self.net[i].loss = 0
        return loss


def gradscale(x, scale):
    """
    x: input tensor
    scale: scale gradient by this
    """
    y_out = x
    y_grad = x * scale
    # y_out returned in forward pass
    # y_grad returned in backward pass
    y = (y_out - y_grad).detach() + y_grad
    return y


def roundpass(x):
    """
    x: input tensor
    """
    y_out = x.round()
    y_grad = x
    # y_out returned in forward pass
    # y_grad returned in backward pass
    y = (y_out - y_grad).detach() + y_grad
    return y


def quanitze(v, s, p, is_activation):
    """
    v: input tensor
    s: step size, learnable
    p: quanitization bits of precision
    is_activation: True if v is activation tensor, False if w is weight tensor
    """
    if is_activation:
        Q_n = 0
        Q_p = 2**p - 1
        grad_scale_factor = 1/torch.sqrt(len(a.view(a.shape[0], -1)) * Q_p)
    else:
        Q_n = - 2**(p-1)
        Q_p = 2**(p-1) - 1
        grad_scale_factor = 1 / torch.sqrt(len(a.view(a.shape[0], -1)))

    s = gradscale(s, grad_scale_factor)
    v = v / s
    v = torch.clip(v, Q_n, Q_p)
    v_bar = roundpass(v)
    v_hat = v_bar * s
    return v_hat
