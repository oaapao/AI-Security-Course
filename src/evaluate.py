# -*- coding: utf-8 -*-
# @File  : evaluate.py
# @Author: oaapao
# @Date  : 2022/9/28
# @Desc  :
# @Contact : zhiqiang.shen@zju.edu.cn
import argparse
import datetime
import os.path

from sklearn.metrics import classification_report

from src import BASE_DIR

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def evaluate(gt_path, pre_path):
    print(f"[{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}]gt path: {gt_path}")
    print(f"[{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}]pre path: {pre_path}")
    assert os.path.exists(gt_path) and os.path.exists(pre_path)
    with open(gt_path, 'r') as f:
        gt_str = f.read()
    with open(pre_path, 'r') as f:
        pre_str = f.read()
    y_true = [classes[int(i)] for i in gt_str.split(' ')[0:-1]]
    y_pred = [classes[int(i)] for i in pre_str.split(' ')[0:-1]]

    with open(os.path.join(task_path, "evaluation.txt"), 'w') as fp:
        fp.write(classification_report(y_true, y_pred))

    print(f"[{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}]Evaluate finished")
    print(
        f"[{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}]Saving evaluation result to: {os.path.join(task_path, 'evaluation.txt')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', help='Task ID', type=int, default=1)
    parser.add_argument('--gt', help='gt label txt path', type=str, default='gt.txt')
    args = parser.parse_args()
    task_path = os.path.join(BASE_DIR, "tasks", f'task-{args.task_id}')

    evaluate(os.path.join(BASE_DIR, args.gt), os.path.join(task_path, 'prediction.txt'))
