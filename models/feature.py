import torch
import torch.nn


class FeatureExtractor(nn.Module):

    """
    This network is used as a feature extractor, takes as input the 'state' defined in
    the AlphaGo Zero paper
    - The state of the past n turns of the board (7 in the paper) for each player.
      This means that the first n matrices of the input state will be 1 and 0, where 1
      is a stone. 
      This is done to take into consideration Go rules (repetitions are forbidden)

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

    def __init__(self, state, ):
        super(FeatureExtractor, self).__init__()
    

    def forward(x):
        return x