import torch.nn as nn
import torch


class ValueNet(nn.Module):

    """
    This network is used to predict which player is more likely to win given the input 'state'
    described in the Feature Extractor model.
    The output is a continuous variable, between -1 and 1. 
    """

    def __init__(self, inplanes):
        super(ValueNet, self).__init__()
        self.conv = nn.Conv2d(inplanes, 2, kernel_size=1)
        self.conv_bn = nn.BatchNorm2d(2)
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 1)
        self.criterion = torch.nn.MSELoss()
        

    def forward(x):
        """
        x : feature maps extracted from the state
        winning : probability of the current agent winning the game
                  considering the actual state of the board
        """
 
        x = F.relu(self.conv_bn(self.conv(x)))
        x = F.relu(self.fc1(x))
        winning = F.tanh(self.fc2(x))
        return winning