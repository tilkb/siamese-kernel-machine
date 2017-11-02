import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 10)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size = 7)
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv2d(128, 128, kernel_size = 4)
        self.dropout3 = nn.Dropout(p=0.1)

        self.conv4 = nn.Conv2d(128, 256, kernel_size = 4)
        self.dropout4 = nn.Dropout(p=0.1)

        self.fc = nn.Linear(9216, 4096)
    

    def forward(self, x):
        block1 = self.dropout1(F.max_pool2d(F.relu(self.conv1(x)),2))
        block2 = self.dropout2(F.max_pool2d(F.relu(self.conv2(block1)),2))
        block3 = self.dropout3(F.max_pool2d(F.relu(self.conv3(block2)),2))
        block4 = self.dropout3(F.relu(self.conv4(block3)))
        flatten = block4.view(-1,9216)
        output = self.fc(flatten)
        return output


class SiameseSVMNet(nn.Module):
    def __init__(self):
        super(SiameseSVMNet, self).__init__()
        self.featureNet = FeatureNet()
        self.fc = nn.Linear(4096, 1)

    def forward(self, x1, x2):
        output1 = self.featureNet(x1)
        output2 = self.featureNet(x2)
        difference = torch.abs(output1 - output2)
        output = self.fc(difference)
        return output

    def get_FeatureNet(self):
        return self.featureNet
         
        