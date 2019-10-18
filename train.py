import torch
import torch.optim as optim
import lib.network as network
import lib.affine as affine
import cv2
from lib.HDF5Image import load_hdf5
from lib.patch_volume import create_patches
import numpy as np

# Workflow


def train_network(input_batch, image):
    net = network.Net()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    optimizer.zero_grad()

    # Forward pass
    pred_theta = net(input_batch)

    predicted_image = affine.affine_transform(image, pred_theta)

    '''
    Now you should calculate some loss connected with the predicted image and backpropagate that loss into the system.

    loss.backward()
    loss_value = loss.item()

    optimizer.step()
    '''

    return predicted_image


if __name__ == '__main__':

    fixed_file = 'J249J70K_proc.h5'
    moving_file = 'J249J70M_proc.h5'

    data, shape = load_hdf5(fixed_file, moving_file)
    image = data.unsqueeze(1)

    stride = 20
    patch_size = 20

    input_batch = create_patches(data, patch_size, stride)
    print(input_batch[0:128, 1:].shape)
    input_batch = input_batch[0:128, 1:]

    warped_image = train_network(input_batch, image)

    warped_image = np.array(warped_image.detach().numpy(), dtype=np.uint8)
    print("Warped image shape: " + str(warped_image.shape))

    cv2.imshow('x', warped_image[0, 0, int(warped_image.shape[2] / 2), :, :])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
