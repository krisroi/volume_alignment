import torch
import torch.nn as nn


class NCC(nn.Module):
    def __init__(self):
        super(NCC, self).__init__()

    def forward(self, fixed_image, warped_image, reduction):
        # Creates a forward pass for the loss function
        return normalized_cross_correlation(fixed_image, warped_image, reduction)


def normalized_cross_correlation(fixed_image, warped_image, reduction='mean'):
    """ Defines a loss after comparing two volumes
    Args:
        warped_image: the transformed image with shape [B, C, D, H, W]
        fixed_image: the image in which to compare to the transformed
            image with shape [B, C, D, H, W]
        reduction: how to reduce the loss-tensor. Default: 'mean' (opt: 'mean', 'sum')
    Returns:
        a reduced single float tensor as defined by either 1 - ncc or -ncc
        (defined in the return statement)
    """

    fixed = (fixed_image[:] - torch.mean(fixed_image, (2, 3, 4), keepdim=True))
    warped = (warped_image[:] - torch.mean(warped_image, (2, 3, 4), keepdim=True))

    fixed_var = torch.sqrt(torch.sum(torch.pow(fixed, 2), (2, 3, 4)))
    warped_var = torch.sqrt(torch.sum(torch.pow(warped, 2), (2, 3, 4)))

    num = torch.sum(torch.mul(fixed, warped), (2, 3, 4))
    den = torch.mul(fixed_var, warped_var)

    alpha = 1.0e-16  # small number to prevent zero-division
    ncc = torch.div(num, (den + alpha))

    if reduction == 'mean':
        ncc = torch.mean(ncc, dim=0, keepdim=True)
    elif reduction == 'sum':
        ncc = torch.sum(ncc, dim=0, keepdim=True)

    return 1 - ncc  # Try experimenting with 1 - NCC and -NCC
