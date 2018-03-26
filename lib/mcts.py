import numpy as np
import random

class Node():

    def __init__(self, move, probas):
        self.p = probas
        self.n = 0
        self.w = 0
        self.q = 0


class MCTS():

    def __init__(self, board, c_puct, value_net, policy_net):
        self.value_net = value_net
        self.policy_net = policy_net
        self.c_puct = c_puct
        self.board = board
    

    def _draw_move(self, action_scores, competitive=False):
        """
        Find the best move, either deterministically for competitive play
        or stochiasticly according to some temperature constant
        """

        if competitive:
            move = np.argmax(action_scores)

        else:
            action_scores = np.power(action_scores, (1. / TEMP))
            total = np.sum(action_scores)
            probas = action_scores / total
            move = np.random.choice(action_scores.shape[0], p=probas)

        return move
    

    def _puct(self, proba, total_count, count):
        """
        Function of P and N that increases if an action hasn't been explored
        much, relative to the other actions, or if the prior probability of the
        action is high, according to the paper
        """
        action_score = self.c_puct * proba 
        action_score *= (np.sqrt(total_count) / (1 + count))
        return action_score


    def select(self, nodes):
        """
        Select the move that maximises the mean value of the next state +
        the result of the PUCT function
        """

        action_scores = []
        total_count = sum([node.visit_count for node in nodes])

        for node in nodes:
            action_score = node.q.mean() + self._puct(node.p, total_count, node.n)
            action_scores.append(action_score)

        return max(action_scores)
    
    def search(self, actions):
        x = random.choice(actions)
        return x


