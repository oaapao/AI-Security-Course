# -*- coding: utf-8 -*-
# @File  : evaluate.py
# @Author: oaapao
# @Date  : 2022/9/28
# @Desc  :
# @Contact : zhiqiang.shen@zju.edu.cn
import argparse
import os.path

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def evaluate(gt_path, pre_path):
    assert os.path.exists(gt_path) and os.path.exists(pre_path)
    with open(gt_path, 'r') as f:
        gt_str = f.read()
    with open(pre_path, 'r') as f:
        pre_str = f.read()
    y_true = [classes[int(i)] for i in gt_str.split(' ')[0:-1]]
    y_pred = [classes[int(i)] for i in pre_str.split(' ')[0:-1]]
    print(classification_report(y_true, y_pred))
    # print(f'Acc. {accuracy_score(y_true, y_pred)}')

    C = confusion_matrix(y_true, y_pred, labels=list(classes))
    plt.matshow(C)
    plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', help='gt label txt path', type=str, default='./result/gt.txt')
    parser.add_argument('--pre', help='predict label txt path', type=str, default='./result/prediction.txt')
    args = parser.parse_args()

    evaluate(args.gt, args.pre)
