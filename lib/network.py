import torch
import torch.nn as nn
from lib.affine import affine_transform


class Net(nn.Module):
    """Model network
        Args:
            x (Tensor): Tensor containing a batch of moving patches
        Returns:
            predicted deformation matrix theta
    """

    def __init__(self):
        super(Net, self).__init__()

        # Spatial transformer localization-network
        self.stn1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=5),
            nn.BatchNorm3d(4, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(4, 16, kernel_size=5),
            nn.BatchNorm3d(16, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(16, 40, kernel_size=5),
            nn.BatchNorm3d(40, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),
            nn.Flatten(),
            nn.Linear(40 * 4 * 4 * 4, 256),
            nn.BatchNorm1d(256, track_running_stats=True),
            nn.ReLU(True),
            nn.Linear(256, 3 * 4)
        )

        self.sampler1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5),
            nn.BatchNorm3d(16, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(16, 32, kernel_size=5),
            nn.BatchNorm3d(32, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2)
        )

        self.stn2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=5),
            nn.BatchNorm3d(32, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4 * 4, 256),
            nn.BatchNorm1d(256, track_running_stats=True),
            nn.ReLU(True),
            nn.Linear(256, 3 * 4)
        )

        self.sampler2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=5),
            nn.BatchNorm3d(64, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2)
        )

        self.stn3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3),
            nn.BatchNorm3d(128, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 1 * 1 * 1, 64),
            nn.BatchNorm1d(64, track_running_stats=True),
            nn.ReLU(True),
            nn.Linear(64, 3 * 4)
        )

        # Initialize the weights/bias with identity transformation
        self.stn1[16].weight.data.zero_()
        self.stn2[8].weight.data.zero_()
        self.stn3[8].weight.data.zero_()
        self.stn1[16].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float64))
        self.stn2[8].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float64))
        self.stn3[8].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float64))

    def forward(self, x):
        """ Performs the usual foward pass through a localization network
        Returns:
            A transformation matrix theta of the shape
            [1, 0, 0, 0]
            [0, 1, 0, 0]
            [0, 0, 1, 0]
        """
        t1 = self.stn1(x)
        t1 = t1.view(-1, 3, 4)
        p1 = affine_transform(x, t1)

        s1 = self.sampler1(p1)
        t2 = self.stn2(s1)
        t2 = t2.view(-1, 3, 4)
        p2 = affine_transform(s1, t2)

        s2 = self.sampler2(p2)
        t3 = self.stn3(s2)

        predicted_theta = t3.view(-1, 3, 4)

        return predicted_theta
