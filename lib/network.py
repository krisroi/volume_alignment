import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 5 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(p=0.2)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(64, 64, kernel_size=3)
        )
        print('======>Loc 1st layer: ', self.localization[0].weight.data.shape)
        print('======>Loc 2nd layer: ', self.localization[3].weight.data.shape)
        print('======>Loc 3rd layer: ', self.localization[6].weight.data.shape)

        # Regressor for the 4*3 matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(8192, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 4 * 3)
        )
        print('======>fc_loc 1st layer: ', self.fc_loc[0].weight.data.shape)
        print('======>fc_loc 2nd layer: ', self.fc_loc[2].weight.data.shape)
        print('======>fc_loc 3rd layer: ', self.fc_loc[4].weight.data.shape)

        # Initialize the weights/bias with identity transformation
        # fc_loc[x] gets the x-th layer in the sequential sequence

        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        """ Performs the usual foward pass through a localization network
        Returns:
            A transformation matrix theta of the shape
            [1, 0, 0, 0]
            [0, 1, 0, 0]
            [0, 0, 1, 0]
        """

        xs = self.localization(x)
        print('Shape after localization network: ', xs.shape)
        xs = xs.view(-1, xs.shape[0] * xs.shape[1] * xs.shape[2] * xs.shape[3] * xs.shape[4])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)
        print(theta)

        return theta

        ''' Old forward pass
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        print("Shape before flat_features: " + str(x.shape))
        x = x.view(-1, self.num_flat_features(x))
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        '''

        # return x
    '''
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    '''


if __name__ == "__main__":
    net = Net()
    print(net)

    input = torch.randn(128, 1, 20, 20, 20)
    out = net(input)

# THE CORRECT THETA IN THIS PROJECT SHOULD INPUT (3, 4)
