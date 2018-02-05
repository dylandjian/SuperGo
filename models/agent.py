import numpy as np
from .policy import PolicyNet
from .value import ValueNet
from const import CUDA



class SuperAgent():

    def __init__(self, inplanes, outplanes):
        if CUDA:
            self.value = ValueNet(inplanes).cuda()
            self.policy = PolicyNet(inplanes, outplanes).cuda()
        else:
            self.value = ValueNet(inplanes)
            self.policy = PolicyNet(inplanes, outplanes)


    def look_ahead(self, state):
        probas = self.policy(state)
        current_best = np.argmax(probas.data.cpu().numpy())
        return current_best