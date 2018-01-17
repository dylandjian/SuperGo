import torch
import torch.nn.functional as F


class ValueNet(nn.Module):

    """
    This network is used to predict which player is more likely to win given the input 'state'
    described in the Feature Extractor model.
    The output is a continuous variable, between -1 and 1. 
    """

    def __init__(self):
        super(ValueNet, self).__init__()
    
    def forward(x):
        return x