import torch
import torch.nn as nn


class NCC(nn.Module):
    def __init__(self):
        super(NCC, self).__init__()

    def forward(self, fixed_image, warped_image, patch=True):
        # Creates a forward pass for the loss function
        return normalized_cross_correlation(fixed_image, warped_image, patch)


def normalized_cross_correlation(fixed_image, warped_image, patch=True):
    """ Defines a loss after comparing two volumes
    Args:
        warped_image: the transformed image
        fixed_image: the image in which to compare to the transformed image
        patch: temporary boolean to decide if the correlation should be done on patches or not (TRUE by default)
    Returns:
        a floating number as defined by either 1 - ncc or -ncc (defined in the return statement)
    """
    if patch:
        fixed_image = fixed_image.unsqueeze(1)  # Adding channel and batch dimension
    else:
        fixed_image = fixed_image.unsqueeze(0).unsqueeze(1)  # Adding channel and batch dimension

    fixed = (fixed_image[:] - torch.mean(fixed_image, (2, 3, 4), keepdim=True))
    warped = (warped_image[:] - torch.mean(warped_image, (2, 3, 4), keepdim=True))

    fixed_var = torch.sqrt(torch.sum(torch.pow(fixed, 2), (2, 3, 4)))
    warped_var = torch.sqrt(torch.sum(torch.pow(warped, 2), (2, 3, 4)))

    num = torch.sum(torch.mul(fixed, warped), (2, 3, 4))
    den = torch.mul(fixed_var, warped_var)

    alpha = 0.0000001
    ncc = torch.div(num, (den + alpha))

    return 1 - ncc  # Try experimenting with 1 - NCC and -NCC
