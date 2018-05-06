import torch.nn as nn
import torch.nn.functional as F
from const import BLOCKS


class BasicBlock(nn.Module):
    """
    Basic residual block with 2 convolutions and a skip connection before the last
    ReLU activation.
    """ 

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out


class Extractor(nn.Module):
    """
    This network is used as a feature extractor, takes as input the 'state' defined in
    the AlphaGo Zero paper
    - The state of the past n turns of the board (7 in the paper) for each player.
      This means that the first n matrices of the input state will be 1 and 0, where 1
      is a stone. 
      This is done to take into consideration Go rules (repetitions are forbidden) and
      give a sense of time

    - The color of the stone that is next to play. This could have been a single bit, but
      for implementation purposes, it is actually expended to the whole matrix size.
      If it is black turn, then the last matrix of the input state will be a NxN matrix
      full of 1, where N is the size of the board, 19 in the case of AlphaGo.
      This is done to take into consideration the komi.

    The ouput is a series of feature maps that retains the meaningful informations
    contained in the input state in order to make a good prediction on both which is more
    likely to win the game from the current state, and also which move is the best one to
    make. 
    """

    def __init__(self, inplanes, outplanes):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=3,
                        padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)

        for block in range(BLOCKS):
            setattr(self, "res{}".format(block), \
                BasicBlock(outplanes, outplanes))
    

    def forward(self, x):
        """
        x : tensor representing the state
        feature_maps : result of the residual layers forward pass
        """

        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(BLOCKS - 1):
            x = getattr(self, "res{}".format(block))(x)
        
        feature_maps = getattr(self, "res{}".format(BLOCKS - 1))(x)
        return feature_maps








