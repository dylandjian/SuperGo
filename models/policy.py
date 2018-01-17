import torch
import torch.nn


class PolicyNet(nn.Module):
    """
    This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state' described in the Feature Extractor model.
    """

    def __init__(self):
        super(PolicyNet, self).__init__()

    def forward(x):
        return x