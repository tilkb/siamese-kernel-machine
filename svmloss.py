import torch
import torch.nn as nn


class SVMLoss(nn.Module):
    def __init__(self, C):
        super(SVMLoss, self).__init__()
        self.C = C

    def forward(self, output, y):
        # loss calculation
        temp1 = (output * y).view(-1)
        ones = temp1 / temp1
        temp2 = ones - temp1
        zeros = temp2 - temp2
        elementwiseloss = torch.max(temp2, zeros)
        elementwiseloss = elementwiseloss * elementwiseloss
        return self.C * torch.mean(elementwiseloss)


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy'''
    predictions = torch.sign(predictions)
    correct = predictions.eq(labels)
    result = correct.sum().data.cpu()
    return result
