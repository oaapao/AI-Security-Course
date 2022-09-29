import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src import BASE_DIR
from src.load_data import classes


def phase_task_result(task_id):
    """
    After evaluate on test result of one task,
    this func help to sort out important info and metrics.
    :param task_id: Train Task ID
    :return: dict
    """
    task_path = os.path.join(BASE_DIR, "tasks", f'task-{task_id}')
    if not os.path.exists(task_path):
        print(f'Task ID {task_id} not exists')
        return {}
    with open(os.path.join(task_path, "task_info.json"), 'r') as f:
        task_info = json.load(f)

    losses = []
    accs = []
    with open(os.path.join(task_path, "train_loss.txt"), "r") as f:
        train_info = [i.split(',') for i in f.read().split('\n') if i]
    for str1, str2 in train_info:
        losses.append(float(str1.split(':')[1].strip()))
        accs.append(float(str2.split(':')[1].strip()[:-1]))

    with open(os.path.join(task_path, "evaluation.txt"), "r") as f:
        evaluation_txt = [i for i in f.read().split('\n')[-2].split(' ') if i]
    acc, recall, f1 = evaluation_txt[3:6]

    return {
        "precision": float(acc),
        "recall": float(recall),
        "f1-score": float(f1),
        "loss": losses,
        "accs": accs,
        "acc": accs[-1],
        "epoch": task_info["epoch"],
        "learning_rate": task_info["learning_rate"],
        "bn": task_info["BN"],
        "dropout": task_info["Dropout"],
    }


def change_learning_rate_analyse(task_ids):
    """
    compare acc
    :param task_ids:
    :return:
    """
    results = []
    for task_id in task_ids:
        results.append(phase_task_result(task_id))
    x_lr = [i["learning_rate"] for i in results]
    y_acc = [i["acc"] for i in results]
    plt.plot(x_lr, y_acc, 'bo--')
    plt.xlabel('learning rate')
    plt.ylabel('Acc. on test set')
    plt.title('Test Accuracy varies with learning rate')
    plt.show()


def change_model_analyse(task_ids):
    """
    compare acc
    :param task_ids:
    :return:
    """
    results = []
    for task_id in task_ids:
        results.append(phase_task_result(task_id))

    size = 4
    acc = [i["acc"] / 100.0 for i in results]
    precisions = [i["precision"] for i in results]
    recall = [i["recall"] for i in results]
    f1_score = [i["f1-score"] for i in results]
    x = np.arange(size)

    total_width, n = 0.7, 4
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.figure(figsize=(15, 6))
    plt.bar(x, acc, width=width, label="Acc")
    plt.bar(x + width, precisions, width=width, label="Precision")
    plt.bar(x + 2 * width, recall, width=width, label="Recall")
    plt.bar(x + 3 * width, f1_score, width=width, label="F1 score")
    x_labels = ["All", "BN", "Dropout", "None"]
    plt.xticks(x, x_labels)
    plt.xlabel('different regularization method')
    plt.ylabel('score')
    plt.title('Test Accuracy varies with regularization method')
    plt.legend(bbox_to_anchor=(1.02, 0.8), loc=3, borderaxespad=0)
    plt.show()


def change_epoch_analyse(task_id):
    """
    Acc and training loss over epochs, confusion matrix
    :param task_id:
    :return:
    """
    result = phase_task_result(task_id)
    x_epoch = range(1, 31)
    y_loss = [i / 5000 for i in result["loss"]]
    y_acc = [i / 100.0 for i in result["accs"]]

    plt.plot(x_epoch, y_loss, label="Training loss")
    plt.plot(x_epoch, y_acc, label="Acc on test set")
    plt.legend()
    plt.xlabel('Epoch')
    plt.title('Test Accuracy and training loss varies with Epoch')
    plt.show()


def show_confusion_matrix(task_id):
    task_path = os.path.join(BASE_DIR, "tasks", f'task-{task_id}')
    with open(os.path.join(BASE_DIR, "gt.txt"), 'r') as f:
        gt_str = f.read()
    with open(os.path.join(task_path, "prediction.txt"), 'r') as f:
        pre_str = f.read()
    y_true = [classes[int(i)] for i in gt_str.split(' ')[0:-1]]
    y_pred = [classes[int(i)] for i in pre_str.split(' ')[0:-1]]

    C = confusion_matrix(y_true, y_pred, labels=classes)
    plt.matshow(C, cmap=plt.cm.Blues)
    plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.draw()
    plt.show()


if __name__ == '__main__':
    lr_tasks = list(range(1, 11))
    model_tasks = list(range(11, 15))
    epoch_task = 11
    # change_learning_rate_analyse(lr_tasks)
    # change_model_analyse(model_tasks)
    # change_epoch_analyse(epoch_task)
    show_confusion_matrix(11)
