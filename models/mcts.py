import numpy as np
import time
from copy import deepcopy
import random
from const import *
from lib.utils import _prepare_state

class Node():

    def __init__(self, parent=None, proba=None, move=None):
        """
        p : probability of reaching that node, given by the policy net
        n : number of time this node has been visited during simulations
        w : total action value, given by the value network
        q : mean action value (w / n)
        """
        self.p = proba
        self.n = 0
        self.w = 0
        self.q = 0
        self.childrens = []
        self.parent = parent
        self.move = move
    
    def update(self, v):
        """ Update the node statistics after a playout """

        self.n = self.n + 1
        self.w = self.w + v
        self.q = self.w / self.n


    def is_leaf(self):
        """ Check whether node is a leaf or not """

        if self.childrens and len(self.childrens) > 0:
            return False
        return True


    def expand(self, probas):
        self.childrens = [Node(parent=self, move=idx, proba=probas[idx]) \
                    for idx in range(probas.shape[0]) if probas[idx] > 0]


class MCTS():

    def __init__(self):
        self.root = Node()


    def _draw_move(self, action_scores, competitive=False):
        """
        Find the best move, either deterministically for competitive play
        or stochiasticly according to some temperature constant
        """

        if competitive:
            move = np.argmax(action_scores)
            total = np.sum(action_scores)
            probas = action_scores / total

        else:
            total = np.sum(action_scores)
            probas = action_scores / total
            move = np.random.choice(action_scores.shape[0], p=probas)

        return move, probas


    @profile
    def _select(self, nodes, c_puct=C_PUCT):
        """
        Select the move that maximises the mean value of the next state +
        the result of the PUCT function
        """

        total_count = sum([node.n for node in nodes])

        sqrt = np.sqrt
        action_scores = np.array([node.q + c_puct * node.p * \
                    (sqrt(total_count) / (1 + node.n)) for node in nodes])
        
        equals = np.where(action_scores == np.max(action_scores))[0]
        if equals.shape[0] > 0:
            return nodes[np.random.choice(equals)]
        return nodes[equals[0]]
    

    def dirichlet_noise(self, probas):
        dim = (probas.shape[0],)
        new_probas = (1 - EPS) * probas + \
                     EPS * np.random.dirichlet(np.full(dim, ALPHA))
        return new_probas


    def advance(self, move):
        print('target: ', move)
        for i in range(len(self.root.childrens)):
            print(self.root.childrens[i].move)
            if self.root.childrens[i].move == move:
                final_idx = i
        self.root = self.root.childrens[final_idx]


    @profile
    def search(self, current_game, player, competitive=False):
        for sim in range(MCTS_SIM):
            game = deepcopy(current_game)
            state = game.state
            current_node = self.root
            done = False

            while not current_node.is_leaf() and not done:
                current_node = self._select(current_node.childrens)
                state, _, done = game.step(current_node.move)

            ## Predict the probas
            if not done:
                state = _prepare_state(state)
                feature_maps = player.extractor(state)

                ## Policy and value prediction
                probas = player.policy_net(feature_maps)
                probas = probas.cpu().data.numpy()[0]
                v = player.value_net(feature_maps)

                ## Add noise in the root node
                if not current_node.parent:
                    probas = self.dirichlet_noise(probas)
                
                ## Modify probability vector depending on valid moves
                valid_moves = game.get_legal_moves()
                illegal_moves = np.setdiff1d(np.arange(game.board_size ** 2 + 1),
                                             np.array(valid_moves))
                probas[illegal_moves] = 0
                total = np.sum(probas)
                probas /= total

                current_node.expand(probas)

                ## Backpropagate the result of the simulation
                while current_node.parent:
                    current_node.update(float(v))
                    current_node = current_node.parent
            else:
                probas = np.zeros((game.board_size ** 2 + 1, ))
                probas[-1] = 1.

        action_scores = np.zeros((game.board_size ** 2 + 1,))
        for node in self.root.childrens:
            action_scores[node.move] = node.n
        final_move, final_probas = self._draw_move(action_scores, competitive=competitive)

        for i in range(len(self.root.childrens)):
            if self.root.childrens[i].move == final_move:
                final_idx = i

        self.root = self.root.childrens[final_idx]
        return final_probas, final_move


