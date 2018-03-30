from .feature import Extractor
from .value import ValueNet
from .policy import PolicyNet
from .mcts import MCTS
from const import *


class Player:

    def __init__(self):
        if CUDA:
            self.extractor = Extractor(INPLANES, OUTPLANES_MAP).cuda()
            self.value_net = ValueNet(OUTPLANES_MAP).cuda()
            self.policy_net = PolicyNet(OUTPLANES_MAP, OUTPLANES).cuda()
        else:
            self.extractor = Extractor(INPLANES, OUTPLANES_MAP)
            self.value_net = ValueNet(OUTPLANES_MAP)
            self.policy_net = PolicyNet(OUTPLANES_MAP, OUTPLANES)    
        self.mcts = MCTS(C_PUCT, self.extractor, self.value_net, self.policy_net)
        self.passed = False
