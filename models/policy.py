import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state' described in the Feature Extractor model.
    """

    def __init__(self, inplanes, outplanes):
        super(PolicyNet, self).__init__()
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc = nn.Linear(outplanes - 1, outplanes)
        

    def forward(self, x):
        """
        x : feature maps extracted from the state
        probas : a NxN + 1 matrix where N is the board size
                 Each value in this matrix represent the likelihood
                 of winning by playing this intersection
        """
 
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1)
        probas = F.log_softmax(self.fc(x), dim=0)
        return probas