import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=2)
        #self.conv3 = nn.Conv3d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        print("Shape before flat_features: " + str(x.shape))
        x = x.view(-1, self.num_flat_features(x))
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    net = Net()
    print(net)
