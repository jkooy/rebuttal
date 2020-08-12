import torch
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct, len(target), correct / len(target)

def Precision(conf_m, num_class):
    precision = []
    for c in range(num_class):
        precision.append(conf_m[c, c] / conf_m[c, :].sum())
    return np.array(precision)


def Recall(conf_m, num_class):
    recall = []
    for c in range(num_class):
        recall.append(conf_m[c, c] / conf_m[:, c].sum())
    return np.array(recall)

def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
