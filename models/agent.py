import numpy as np
from .policy import PolicyNet
from .value import ValueNet



class SuperAgent():

    def __init__(self, inplanes, outplanes):
        self.value = ValueNet(inplanes).cuda()
        self.policy = PolicyNet(inplanes, outplanes).cuda()


    def look_ahead(self, state):
        probas = self.policy(state)
        current_best = np.argmax(probas.data.cpu().numpy())
        return current_best