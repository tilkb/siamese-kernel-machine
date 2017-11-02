import torch
import torch.nn as nn

class SVMLoss(nn.Module):
    def __init__(self, C):
        super(SVMLoss, self).__init__()
        self.C = C

    def forward(self, output, y):
        # loss calculation
        temp1 = (output * y).view(-1)
        temp2 = temp1 / temp1 - temp1  # TODO:one/zero tensor
        zeros = temp2 - temp2
        elementwiseloss = torch.max(temp2, zeros)
        elementwiseloss = elementwiseloss * elementwiseloss
        return self.C * torch.mean(elementwiseloss)

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    predictions = torch.sign(predictions)
    correct = predictions.eq(labels)
    result = correct.sum().data.cpu()
    return result
