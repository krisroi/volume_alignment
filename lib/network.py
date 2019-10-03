import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils as ut
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
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Max pooling over a (2, 2, 2) window
        print("Input shape: " + str(x.shape))
        ut.show_single(x[0, :].detach().numpy(), x[0, :].detach().numpy().shape)
        x = self.act(self.conv1(x))
        ut.show_single(x[0, :].detach().numpy(), x[0, :].detach().numpy().shape)
        print("Shape before 1st pool: " + str(x.shape))
        x = self.pool(x)
        ut.show_single(x[0, :].detach().numpy(), x[0, :].detach().numpy().shape)
        print("Shape before 2nd conv: " + str(x.shape))
        x = self.act(self.conv2(x))
        ut.show_single(x[0, :].detach().numpy(), x[0, :].detach().numpy().shape)
        print("Shape before 2nd pool: " + str(x.shape))
        x = self.pool(x)
        ut.show_single(x[0, :].detach().numpy(), x[0, :].detach().numpy().shape)
        #print("Shape before 3rd conv: " + str(x.shape))
        #x = self.act(self.conv3(x))
        #ut.show_single(x[0, :].detach().numpy(), x[0, :].detach().numpy().shape)
        #print("Shape before 3rd pool: " + str(x.shape))
        #x = self.pool(x)
        print("Shape before flat_features: " + str(x.shape))
        x = x.view(-1, self.num_flat_features(x))
        print("Shape before first fully connected: " + str(x.shape))
        x = self.act(self.fc1(x))
        print("Shape before second fully connected: " + str(x.shape))
        #x = self.dropout(x)
        x = self.act(self.fc2(x))
        print("Shape before output: " + str(x.shape))
        x = self.fc3(x)
        print("Shape at output: " + str(x.shape))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
