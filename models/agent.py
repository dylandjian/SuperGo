from .feature import Extractor
from .value import ValueNet
from .policy import PolicyNet
import os
from .mcts import MCTS
from const import *


class Player:

    def __init__(self):
        if CUDA:
            self.extractor = Extractor(INPLANES, OUTPLANES_MAP).cuda()
            self.value_net = ValueNet(OUTPLANES_MAP, OUTPLANES).cuda()
            self.policy_net = PolicyNet(OUTPLANES_MAP, OUTPLANES).cuda()
        else:
            self.extractor = Extractor(INPLANES, OUTPLANES_MAP)
            self.value_net = ValueNet(OUTPLANES_MAP, OUTPLANES)
            self.policy_net = PolicyNet(OUTPLANES_MAP, OUTPLANES)    
        self.mcts = MCTS(C_PUCT, self.extractor, self.value_net, self.policy_net)
        self.passed = False
    
    def save_models(self, improvements, current_time, optimizer=None):
        for model in ["extractor", "policy_net", "value_net"]:
            self._save_checkpoint(getattr(self, model),\
                                improvements, model, current_time, optimizer)

    def _save_checkpoint(self, model, current_version, filename, current_time, optimizer):
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                            '..', 'saved_models', current_time)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        state = {
            'state_dict': model.state_dict(),
            'version': current_version,
        }
        if optimizer:
            state['optimizer'] = optimizer.state_dict(),
        filename = os.path.join(dir_path, "{}-{}.pth.tar".format(current_version, filename))
        torch.save(state, filename)

    def load_models(self, path, models):
        names = ["extractor", "policy_net", "value_net"]
        for i in range(0, len(models)):
            checkpoint = torch.load(os.path.join(path, models[i]))
            model = getattr(self, names[i])
            model.load_state_dict(checkpoint["state_dict"])

