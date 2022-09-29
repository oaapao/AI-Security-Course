# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: oaapao
# @Date  : 2022/9/28
# @Desc  : train model
# @Contact : zhiqiang.shen@zju.edu.cn
import argparse
import datetime
import os

import torch
import torch.optim as optim
from torch import nn
from tqdm.auto import trange

from load_data import trainloader
from model import Net
from src import BASE_DIR


def train(epochs):
    print('Start Training')
    for epoch in trange(epochs, desc='Epoch'):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f'Epoch {epoch + 1} total loss: {running_loss:.3f}')
        torch.save(net.state_dict(), PATH.format(epoch + 1))

    print('Finished Training')
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', help='train task ID', type=int, default=1)
    parser.add_argument('--epoch', help='train epoch', type=int, default=5)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.0003)
    parser.add_argument('--device', help='cpu or gpu', type=str, default='cuda:3')
    args = parser.parse_args()

    device = torch.device(args.device)
    net = Net()
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    cur_time = datetime.datetime.now()
    date_str = cur_time.strftime('%m%d-')
    time_str = cur_time.strftime('%H%M')
    path = os.path.join(BASE_DIR, "result", f'task-{args.task_id}', "weights")

    if os.path.exists(path):
        raise RuntimeError(f"Task ID {args.task_id} already exists.")
    os.makedirs(path)

    # TODO: write task info

    PATH = os.path.join(path, '{}.pth')
    train(args.epoch)
