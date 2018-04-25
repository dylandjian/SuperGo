import numpy as np
import threading
import time
import random
from numba import jit
from copy import deepcopy
from const import *
from lib.utils import _prepare_state, sample_rotation


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
        self.childrens = [Node(parent=self, move=idx, proba=probas[idx]) \
                    for idx in range(probas.shape[0]) if probas[idx] > 0]


def dirichlet_noise(probas):
    """ Add Dirichlet noise in the root node """

    dim = (probas.shape[0],)
    new_probas = (1 - EPS) * probas + \
                    EPS * np.random.dirichlet(np.full(dim, ALPHA))
    return new_probas


def _select(nodes, c_puct=C_PUCT):
    """
    Select the move that maximises the mean value of the next state +
    the result of the PUCT function
    """

    return nodes[_opt_select(np.array([[node.q, node.n, node.p] \
                    for node in nodes]), c_puct)]

@jit
def _opt_select(nodes, c_puct):
    """ Optimized version of the selection """

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


class EvaluatorThread(threading.Thread):
    def __init__(self, player, eval_queue, result_queue, condition_search, condition_eval, condition_mix):
        threading.Thread.__init__(self)
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.player = player
        self.condition_search = condition_search
        self.condition_eval = condition_eval
        self.condition_mix = condition_mix

    def run(self):
        total_sim = MCTS_SIM
        while total_sim > 0:
            self.condition_eval.acquire()
            while (len(self.eval_queue.values()) < BATCH_SIZE_EVAL or \
                  len(self.result_queue.values()) > 0):
                print("notifying in evaluator, current len: %d" % len(self.eval_queue.values()))
                self.condition_eval.wait()
            self.condition_eval.release()

            states = []
            for idx, state in self.eval_queue.items():
                states.append(sample_rotation(state, num=1))
            print('states len: %d' % len(states))
            states = _prepare_state(states)
            feature_maps = self.player.extractor(states[0])

            ## Policy and value prediction
            probas = self.player.policy_net(feature_maps)
            v = self.player.value_net(feature_maps)
            keys = list(self.eval_queue.keys())
            idx = 0
            for key in keys:
                self.result_queue[key] = (probas[idx].cpu().data.numpy(), float(v[idx]))
                del self.eval_queue[key]
                idx += 1
            del probas, v, feature_maps
            self.condition_mix.acquire()
            self.condition_mix.notifyAll()
            self.condition_mix.release()
            total_sim -= BATCH_SIZE_EVAL



class SearchThread(threading.Thread):
    def __init__(self, mcts, game, eval_queue, result_queue, thread_id, \
                lock, condition_search, condition_eval, condition_mix):
        threading.Thread.__init__(self)
        self.result_queue = result_queue
        self.eval_queue = eval_queue
        self.mcts = mcts
        self.game = game
        self.lock = lock
        self.thread_id = thread_id
        self.condition_eval = condition_eval
        self.condition_search = condition_search
        self.condition_mix = condition_mix
    

    def run(self):
        game = deepcopy(self.game)
        state = game.state
        current_node = self.mcts.root
        done = False

        while not current_node.is_leaf() and not done:
            current_node = _select(current_node.childrens)
            self.lock.acquire()
            current_node.n += 1
            self.lock.release()
            state, _, done = game.step(current_node.move)

        ## Predict the probas
        if not done:
            self.condition_search.acquire()
            while len(self.eval_queue.values()) < BATCH_SIZE_EVAL and \
                  len(self.result_queue.values()) == 0:
                print("trying to release in thread id %d" % self.thread_id)
                self.condition_search.wait()
            print("added move in thread %d" % self.thread_id)
            self.condition_search.release()
            
            self.eval_queue[self.thread_id] = state
            self.condition_eval.acquire()
            self.condition_eval.notify()
            self.condition_eval.release()

            self.condition_mix.acquire()
            self.condition_mix.wait()
            self.condition_mix.release()

            probas = np.array(self.result_queue[self.thread_id][0], copy=True)
            v = float(self.result_queue[self.thread_id][1])
            del self.result_queue[self.thread_id]

            ## Add noise in the root node
            if not current_node.parent:
                probas = dirichlet_noise(probas)
            
            ## Modify probability vector depending on valid moves
            valid_moves = game.get_legal_moves()
            illegal_moves = np.setdiff1d(np.arange(game.board_size ** 2 + 1),
                                        np.array(valid_moves))
            probas[illegal_moves] = 0
            total = np.sum(probas)
            probas /= total

            self.lock.acquire()
            current_node.expand(probas)

            ## Backpropagate the result of the simulation
            while current_node.parent:
                current_node.update(v)
                current_node = current_node.parent
            self.lock.release()
            print("done in thread id %d" % self.thread_id)


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
        threads = []
        condition_eval = threading.Condition()
        condition_search = threading.Condition()
        condition_mix = threading.Condition()
        lock = threading.Lock()
        eval_queue = {}
        result_queue = {}
        evaluator = EvaluatorThread(player, eval_queue, result_queue, condition_search, \
                                    condition_eval, condition_mix)
        evaluator.start()
        for sim in range(MCTS_SIM // MCTS_PARALLEL):
            eval_queue.clear()
            for idx in range(MCTS_PARALLEL):
                threads.append(SearchThread(self, current_game, eval_queue, result_queue, idx, 
                                        lock, condition_search, condition_eval, condition_mix))
                threads[-1].start()
            for thread in threads:
                thread.join()
        evaluator.join()

        action_scores = np.zeros((current_game.board_size ** 2 + 1,))
        for node in self.root.childrens:
            action_scores[node.move] = node.n
        final_move, final_probas = self._draw_move(action_scores, competitive=competitive)

        for idx in range(len(self.root.childrens)):
            if self.root.childrens[idx].move == final_move:
                break

        self.root = self.root.childrens[idx]
        print(final_probas, final_move)
        return final_probas, final_move


