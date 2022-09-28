# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: oaapao
# @Date  : 2022/9/28
# @Desc  : define classify model on CIFAR10
# @Contact : zhiqiang.shen@zju.edu.cn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size)
        )

    def forward(self, x):
        return x * self.se(self.pool(x))


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


# class Net(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.conv0 = nn.Conv2d(3, 8, 3, padding=1)
#         self.bn0 = nn.BatchNorm2d(8)
#         self.dw1 = DepthWiseConv(8, 16)
#         self.down1 = DepthWiseConv(16, 32, 2)
#         self.bn1_0 = nn.BatchNorm2d(16)
#         self.bn1_1 = nn.BatchNorm2d(32)
#         self.se1 = SeModule(32)
#
#         self.dw2 = DepthWiseConv(32, 32)
#         self.down2 = DepthWiseConv(32, 64, 2)
#         self.bn2_0 = nn.BatchNorm2d(32)
#         self.bn2_1 = nn.BatchNorm2d(64)
#         self.se2 = SeModule(64)
#
#         self.fc1 = nn.Linear(64 * 8 * 8, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#         self.dropout = nn.Dropout(p=0.5)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         x = F.relu(self.bn0(self.conv0(x)))
#
#         x = F.relu(self.bn1_0(self.dw1(x)))
#         x = self.se1(F.relu(self.bn1_1(self.down1(x))))
#
#         x = F.relu(self.bn2_0(self.dw2(x)))
#         x = self.se2(F.relu(self.bn2_1(self.down2(x))))
#
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.dropout(self.fc1(x)))
#         x = F.relu(self.dropout(self.fc2(x)))
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 这次使用的训练集是一个只有十个分类的 分类任务  所以这次就是10

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3,32,32) output(16,28,28)
        x = self.pool1(x)  # output(16，14，14)经过最大池化高度和宽度减为原来一半
        x = F.relu(self.conv2(x))  # output(32，10，10)
        x = self.pool2(x)  # output(32，5，5)
        x = x.view(-1, 32 * 5 * 5)  # 进行展平操作 output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x


if __name__ == '__main__':
    net = Net()
    with torch.no_grad():
        a = torch.FloatTensor(1, 3, 32, 32)
        net.eval()
        y = net(a)
        print(y.shape)
