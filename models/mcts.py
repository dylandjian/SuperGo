import numpy as np
import torch
import threading
import time
import random
from collections import OrderedDict
from numba import jit
from copy import deepcopy
from const import *
from lib.utils import _prepare_state, sample_rotation


@jit
def _opt_select(nodes, c_puct=C_PUCT):
    """ Optimized version of the selection based of the PUCT formula """

    total_count = 0
    for i in range(nodes.shape[0]):
        total_count += nodes[i][1]
    
    action_scores = np.zeros(nodes.shape[0])
    for i in range(nodes.shape[0]):
        action_scores[i] = nodes[i][0] + c_puct * nodes[i][2] * \
                (np.sqrt(total_count) / (1 + nodes[i][1]))
    
    equals = np.where(action_scores == np.max(action_scores))[0]
    if equals.shape[0] > 0:
        return np.random.choice(equals)
    return equals[0]


def dirichlet_noise(probas):
    """ Add Dirichlet noise in the root node """

    dim = (probas.shape[0],)
    new_probas = (1 - EPS) * probas + \
                    EPS * np.random.dirichlet(np.full(dim, ALPHA))
    return new_probas


class Node:

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

        self.w = self.w + v
        self.q = self.w / self.n if self.n > 0 else 0


    def is_leaf(self):
        """ Check whether node is a leaf or not """

        return len(self.childrens) == 0


    def expand(self, probas):
        """ Create a child node for every non-zero move probability """

        self.childrens = [Node(parent=self, move=idx, proba=probas[idx]) \
                    for idx in range(probas.shape[0]) if probas[idx] > 0]



class EvaluatorThread(threading.Thread):
    def __init__(self, player, eval_queue, result_queue, condition_search, condition_eval):
        """ Used to be able to batch evaluate positions during tree search """

        threading.Thread.__init__(self)
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.player = player
        self.condition_search = condition_search
        self.condition_eval = condition_eval


    def run(self):
        for sim in range(MCTS_SIM // MCTS_PARALLEL):

            ## Wait for the eval_queue to be filled by new positions to evaluate
            self.condition_search.acquire()
            while len(self.eval_queue) < MCTS_PARALLEL:
                self.condition_search.wait()
            self.condition_search.release()

            self.condition_eval.acquire()
            while len(self.result_queue) < MCTS_PARALLEL:
                keys = list(self.eval_queue.keys())
                max_len = BATCH_SIZE_EVAL if len(keys) > BATCH_SIZE_EVAL else len(keys)

                ## Predict the feature_maps, policy and value
                states = torch.tensor(np.array(list(self.eval_queue.values()))[0:max_len],
                            dtype=torch.float, device=DEVICE)
                v, probas = self.player.predict(states)

                ## Replace the state with the result in the eval_queue
                ## and notify all the threads that the result are available
                for idx, i in zip(keys, range(max_len)):
                    del self.eval_queue[idx]
                    self.result_queue[idx] = (probas[i].cpu().data.numpy(), v[i])

                self.condition_eval.notifyAll()
            self.condition_eval.release()



class SearchThread(threading.Thread):

    def __init__(self, mcts, game, eval_queue, result_queue, thread_id, lock, condition_search, condition_eval):
        """ Run a single simulation """

        threading.Thread.__init__(self)
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.mcts = mcts
        self.game = game
        self.lock = lock
        self.thread_id = thread_id
        self.condition_eval = condition_eval
        self.condition_search = condition_search
    

    def run(self):
        game = deepcopy(self.game)
        state = game.state
        current_node = self.mcts.root
        done = False

        ## Traverse the tree until leaf
        while not current_node.is_leaf() and not done:
            ## Select the action that maximizes the PUCT algorithm
            current_node = current_node.childrens[_opt_select( \
                    np.array([[node.q, node.n, node.p] \
                    for node in current_node.childrens]))]

            ## Virtual loss since multithreading
            self.lock.acquire()
            current_node.n += 1
            self.lock.release()

            state, _, done = game.step(current_node.move)

        if not done:

            ## Add current leaf state with random dihedral transformation
            ## to the evaluation queue
            self.condition_search.acquire()
            self.eval_queue[self.thread_id] = sample_rotation(state, num=1)
            self.condition_search.notify()
            self.condition_search.release()

            ## Wait for the evaluator to be done
            self.condition_eval.acquire()
            while self.thread_id not in self.result_queue.keys():
                self.condition_eval.wait()

            ## Copy the result to avoid GPU memory leak
            result = self.result_queue.pop(self.thread_id)
            probas = np.array(result[0])
            v = float(result[1])
            self.condition_eval.release()

            ## Add noise in the root node
            if not current_node.parent:
                probas = dirichlet_noise(probas)
            
            ## Modify probability vector depending on valid moves
            ## and normalize after that
            valid_moves = game.get_legal_moves()
            illegal_moves = np.setdiff1d(np.arange(game.board_size ** 2 + 1),
                                        np.array(valid_moves))
            probas[illegal_moves] = 0
            total = np.sum(probas)
            probas /= total

            ## Create the child nodes for the current leaf
            self.lock.acquire()
            current_node.expand(probas)

            ## Backpropagate the result of the simulation
            while current_node.parent:
                current_node.update(v)
                current_node = current_node.parent
            self.lock.release()



class MCTS:
    def __init__(self):
        self.root = Node()


    def _draw_move(self, action_scores, competitive=False):
        """
        Find the best move, either deterministically for competitive play
        or stochiasticly according to some temperature constant
        """

        if competitive:
            moves = np.where(action_scores == np.max(action_scores))[0]
            move = np.random.choice(moves)
            total = np.sum(action_scores)
            probas = action_scores / total

        else:
            total = np.sum(action_scores)
            probas = action_scores / total
            move = np.random.choice(action_scores.shape[0], p=probas)

        return move, probas


    def advance(self, move):
        """ Manually advance in the tree, used for GTP """

        for idx in range(len(self.root.childrens)):
            if self.root.childrens[idx].move == move:
                final_idx = idx
                break
        self.root = self.root.childrens[final_idx]


    def search(self, current_game, player, competitive=False):
        """
        Search the best moves through the game tree with
        the policy and value network to update node statistics
        """

        ## Locking for thread synchronization
        condition_eval = threading.Condition()
        condition_search = threading.Condition()
        lock = threading.Lock()

        ## Single thread for the evaluator (for now)
        eval_queue = OrderedDict()
        result_queue = {}
        evaluator = EvaluatorThread(player, eval_queue, result_queue, condition_search, condition_eval)
        evaluator.start()

        threads = []
        ## Do exactly the required number of simulation per thread
        for sim in range(MCTS_SIM // MCTS_PARALLEL):
            for idx in range(MCTS_PARALLEL):
                threads.append(SearchThread(self, current_game, eval_queue, result_queue, idx, 
                                        lock, condition_search, condition_eval))
                threads[-1].start()
            for thread in threads:
                thread.join()
        evaluator.join()

        ## Create the visit count vector
        action_scores = np.zeros((current_game.board_size ** 2 + 1,))
        for node in self.root.childrens:
            action_scores[node.move] = node.n

        ## Pick the best move
        final_move, final_probas = self._draw_move(action_scores, competitive=competitive)

        ## Advance the root to keep the statistics of the childrens
        for idx in range(len(self.root.childrens)):
            if self.root.childrens[idx].move == final_move:
                break
        self.root = self.root.childrens[idx]

        return final_probas, final_move


