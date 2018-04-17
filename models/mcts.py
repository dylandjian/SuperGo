import numpy as np
from copy import deepcopy
import random
from const import *
from lib.utils import _prepare_state

class Node():

    def __init__(self, parent=None, probas=None):
        """
        p : probability of reaching that node, given by the policy net
        n : number of time this node has been visited during simulations
        w : total action value, given by the value network
        q : mean action value (w / n)
        """
        self.p = probas
        self.n = 0
        self.w = 0
        self.q = 0
        self.childrens = None
        self.move = None
    

    def update(self, v):
        """ Update the node statistics after a playout """

        self.n += 1
        self.w += v
        self.q = self.w / self.n


    def is_leaf(self):
        """ Check whether node is a leaf or not """

        if self.childrens and len(self.childrens) > 0:
            return False
        return True


    def expand(self):
        pass


class MCTS():

    def __init__(self, player, competitive=False):
        self.root = Node()
        self.player = player
        self.temp = TEMP
        self.competitive = competitive


    def _draw_move(self, action_scores):
        """
        Find the best move, either deterministically for competitive play
        or stochiasticly according to some temperature constant
        """

        if self.competitive:
            move = np.argmax(action_scores)

        else:
            action_scores = np.power(action_scores, (1. / self.temp))
            total = np.sum(action_scores)
            probas = action_scores / total
            move = np.random.choice(action_scores.shape[0], p=probas)

        return move


    def _puct(self, proba, total_count, count, c_puct=C_PUCT):
        """
        Function of P and N that increases if an action hasn't been explored
        much, relative to the other actions, or if the prior probability of the
        action is high, according to the paper
        """
        action_score = c_puct * proba 
        action_score *= (np.sqrt(total_count) / (1 + count))
        return action_score


    def _select(self, nodes):
        """
        Select the move that maximises the mean value of the next state +
        the result of the PUCT function
        """

        action_scores = []
        total_count = sum([node.visit_count for node in nodes])

        for node in nodes:
            action_score = node.q + self._puct(node.p, total_count, node.n)
            action_scores.append(action_score)

        return max(action_scores)
    

    def dirichlet_noise(self, probas):
        probas = probas.cpu().data.numpy()
        dim = (probas.shape[1],)
        new_probas = (1 - EPS) * probas + \
                     EPS * np.random.dirichlet(np.full(dim, ALPHA))
        return new_probas



    def search(self, current_game, competitive=False):
        for sim in range(MCTS_SIM):
            game = deepcopy(current_game)
            state = game.state
            current_node = self.root

            mov = 0
            while not current_node.is_leaf():
                current_node = self._select(current_node.childrens)
                state, _, _ = game.step(current_node.move)
                mov += 1
            
            state = _prepare_state(state)
            probas = self.player.policy_net(self.player.extractor(state))

            if mov == 0:
                probas = self.dirichlet_noise(probas)
            
            valid_moves = current_game.get_legal_moves()
            probas[np.setdiff1d(np.arange(probas.shape[0]), np.array(valid_moves))] = 0
            total = np.sum(probas)
            probas /= total

        return self._draw_move(action_scores)


