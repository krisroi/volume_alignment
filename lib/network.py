import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=5),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(64, 128, kernel_size=3),
            nn.BatchNorm3d(128),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 4 matrix
        # Numbers in the input of the linear layers are
        #   64 = number of feature maps from the last convolution in localization
        #   x * x * x = D * H * W out of the localization network
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 2 * 2 * 2, 128),
            nn.ReLU(True),
            #nn.Dropout(p=0.2),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 4)
        )

        # Initialize the weights/bias with identity transformation
        # fc_loc[x] gets the x-th layer in the sequential sequence
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float64))

    def forward(self, x):
        """ Performs the usual foward pass through a localization network
        Returns:
            A transformation matrix theta of the shape
            [1, 0, 0, 0]
            [0, 1, 0, 0]
            [0, 0, 1, 0]
        """
        xs = self.localization(x)
        xs = xs.view(-1, xs.shape[1] * xs.shape[2] * xs.shape[3] * xs.shape[4])

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)

        return theta
