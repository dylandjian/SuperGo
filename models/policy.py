import torch
import torch.nn


class PolicyNet(nn.Module):
    """
    This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state' described in the Feature Extractor model.
    """

    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv = nn.Conv2d(128, 4, kernel_size=1)
        self.conv_bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, BOARD_SIZE ** 2 + 1)
        

    def forward(x):
        """
        x : feature maps extracted from the state
        probas : a NxN + 1 matrix where N is the board size
                 Each value in this matrix represent the likelihood
                 of winning by playing this intersection
        """
 
        x = F.relu(self.conv_bn(self.conv(x)))
        probas = F.log_softmax(self.fc(x))
        return probas