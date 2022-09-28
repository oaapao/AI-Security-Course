# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: oaapao
# @Date  : 2022/9/28
# @Desc  : run classify model to get output
# @Contact : zhiqiang.shen@zju.edu.cn
import argparse

import torch
from tqdm import tqdm

from load_data import testloader
from model import Net


def run_test(model_path, device):
    net = Net()
    net.load_state_dict(torch.load(model_path))
    device = torch.device(device)
    net.to(device)
    net.eval()

    correct = 0
    total = 0
    result = ''
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.cpu().size(0)
            correct += (predicted == labels.cpu()).sum().item()

            result += ' '.join(f'{j}' for j in predicted)
            result += ' '

    with open('./result/prediction.txt', 'w') as fp:
        fp.write(result)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='which weight file to load', type=str,
                        default='./weights/0928-2103/20.pth')
    parser.add_argument('--device', help='cpu or gpu', type=str, default='cuda:3')
    args = parser.parse_args()
    run_test(args.path, args.device)
