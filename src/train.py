# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: oaapao
# @Date  : 2022/9/28
# @Desc  : train model
# @Contact : zhiqiang.shen@zju.edu.cn
import argparse
import datetime
import json
import os

import torch
import torch.optim as optim
from torch import nn
from tqdm.auto import trange

from load_data import trainloader, testloader
from model import Net
from src import BASE_DIR
from src.utils import update_json


def train(epochs):
    print('Start Training')
    try:
        for epoch in trange(epochs, desc='Epoch'):  # loop over the dataset multiple times
            net.train()
            running_loss = 0.0
            total = 0
            correct = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            with torch.no_grad():
                net.eval()
                for data in testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.cpu().data, 1)
                    total += labels.cpu().size(0)
                    correct += (predicted == labels.cpu()).sum().item()
            # logging
            line = f'Epoch {epoch + 1}  total loss: {running_loss:.2f}, Acc. on test set: {(100 * correct / total):.2f}%\n'
            with open(os.path.join(task_path, "train_loss.txt"), 'a+') as f:
                f.write(line)

            torch.save(net.state_dict(), PATH.format(epoch + 1))
    except Exception as e:
        update_json(task_info_path, "traceback", e.__str__())

    print('Finished Training')
    update_json(task_info_path, "end_train_time", datetime.datetime.now().strftime('%m.%d %H:%M'))


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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    task_path = os.path.join(BASE_DIR, "tasks", f'task-{args.task_id}')
    print(f"Saving task result to: {task_path}")

    weights_path = os.path.join(task_path, "weights")

    if os.path.exists(weights_path):
        raise RuntimeError(f"Task ID {args.task_id} already exists.")
    os.makedirs(weights_path)
    task_info_path = os.path.join(task_path, "task_info.json")
    with open(task_info_path, "w") as fp:
        json.dump({"start_train_time": datetime.datetime.now().strftime('%m.%d %H:%M'),
                   "epoch": args.epoch,
                   "learning_rate": args.lr}, fp)

    PATH = os.path.join(weights_path, '{}.pth')
    train(args.epoch)
