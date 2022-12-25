#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torchvision
from model import LeNet

EPOCHS = 32
WARMUP = 3
BATCH_SIZE = 128
LR = 0.1
LAMDA = 0.05

model = t
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# loss = cross entropy loss + bin regularization loss
# cross entropy loss -> updates weights of conv2d and linear params
# bin regularization loss -> updates bin size

for epoch in range(EPOCHS):
    for x, y in loader_train:
        # move to device
        x = x.to(device)
        y = y.to(device)
        # forward pass
        y_pred = model(x)
        # br loss
        loss_br = model.loss()  # bin regularization loss
        # cross entropy loss
        if epoch < WARMUP:
            loss_ce = 0
        else:
            loss_ce = nn.CrossEntropyLoss(y_pred, y)

        # backward pass
        loss = loss_ce + LAMDA * loss_br
        loss.backward()
        optimizer.step()
