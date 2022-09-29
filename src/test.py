# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: oaapao
# @Date  : 2022/9/28
# @Desc  : run classify model to get predicts
# @Contact : zhiqiang.shen@zju.edu.cn
import argparse
import datetime
import os

import torch
from tqdm import tqdm

from load_data import testloader
from model import get_model
from src import BASE_DIR


def run_test(model_path, device):
    print(f"[{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}]Tests start")
    try:
        assert os.path.exists(model_path)
        net.load_state_dict(torch.load(model_path))
        device = torch.device(device)
        net.to(device)
        net.eval()

        result = ''
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                images = images.to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.cpu().data, 1)
                result += ' '.join(f'{j}' for j in predicted)
                result += ' '
        # save test results
        log_path = os.path.join(task_path, 'prediction.txt')
        with open(log_path, 'w') as fp:
            fp.write(result)
        print(f"[{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}]Test finished")
        print(
            f"[{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}]Saving result to: {log_path}")
    except Exception as e:
        print(f"[{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}]Test failed")
        with open(os.path.join(task_path, 'prediction.txt'), 'w') as fp:
            fp.write(e.__str__())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', help='Task ID', type=int, default=1)
    parser.add_argument('--epoch', help='which epoch result to use', type=int, default=5)
    parser.add_argument('--device', help='cpu or gpu', type=str, default='cuda:3')
    parser.add_argument('--no_dropout', action='store_true', default=False)
    parser.add_argument('--no_bn', action='store_true', default=False)

    args = parser.parse_args()
    net = get_model(dropout=(not args.no_dropout), bn=(not args.no_bn))
    task_path = os.path.join(BASE_DIR, "tasks", f'task-{args.task_id}')
    run_test(os.path.join(task_path, "weights", f"{args.epoch}.pth"), args.device)
