# -*- coding: utf-8 -*-
# @File  : load_data.py
# @Author: oaapao
# @Date  : 2022/9/28
# @Desc  : load CIFAR10 data
# @Contact : zhiqiang.shen@zju.edu.cn
import os

import torch
import torchvision
import torchvision.transforms as transforms

from src import BASE_DIR

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 24

trainset = torchvision.datasets.CIFAR10(root=os.path.join(BASE_DIR, "data"), train=True, download=False,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root=os.path.join(BASE_DIR, "data"), train=False, download=False,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def make_gt_txt():
    a = ''
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        a += ' '.join(f'{j}' for j in labels)
        a += ' '
    with open(os.path.join(BASE_DIR, "src", "gt.txt"), 'w') as fp:
        fp.write(a)


if __name__ == '__main__':
    # make_gt_txt()
    import matplotlib.pyplot as plt
    import numpy as np


    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
