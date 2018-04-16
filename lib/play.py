import multiprocessing
import timeit
import random
import numpy as np
import pickle
import time
import os
from copy import deepcopy
from models.agent import Player
from .go import GoEnv as Board
from const import *
from pymongo import MongoClient
from torch.autograd import Variable
from .utils import get_player, load_player


def add_games(queue, player, opponent, match_number, cores):
    """ Add tasks to the queue to be ran in parallel """

    for id in range(match_number):
        queue.put(Game(player, id, opponent=opponent))
    
    for _ in range(cores):
        queue.put(None)


def create_matches(player, opponent=None, cores=1, match_number=10):
    """ Create the process queue """

    queue = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    game_results = []

    game_managers = [
        GameManager(queue, results)
        for _ in range(cores)
    ]

    for game_manager in game_managers:
        game_manager.start()

    add_games(queue, player, opponent, match_number, cores)
    
    return queue, results



def self_play(current_time, ite):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    client = MongoClient()
    collection = client.superGo[current_time]
    game_id = 0
    improvements = 1
    player = False

    while True:

        ## Load the player when restarting traning
        if ite:
            new_player, improvements = load_player(current_time, ite)
            game_id = collection.find().count()
            ite = False
        else:
            new_player, improvements = get_player(current_time, improvements)

        print("[PLAY] Current improvement level: %d" % improvements)
        if improvements == 1 and not player and not new_player:
            print("[PLAY] Waiting for first player")
            time.sleep(5)
            continue

        if new_player:
            player = new_player
            print("[PLAY] New player !")

        queue, results = create_matches(player , \
                    cores=PARRALEL_SELF_PLAY, match_number=SELF_PLAY_MATCH) 
        print("[PLAY] Starting to fetch fresh games")
        for _ in range(SELF_PLAY_MATCH):
            result = results.get()
            if result:
                collection.insert({
                    "game": result,
                    "id": game_id
                })
                game_id += 1
        print("[PLAY] Done fetching")
        queue.close()
        time.sleep(15)


def play(player, opponent):
    """ Game between two players, for evaluation """

    queue, results = create_matches(deepcopy(player), opponent=deepcopy(opponent), \
                cores=PARRALEL_EVAL, match_number=EVAL_MATCHS) 
    queue.join()
    
    print("[EVALUATION] Starting to fetch fresh games")
    final_result = []
    for _ in range(EVAL_MATCHS):
        result = results.get()
        if result:
            final_result.append(pickle.loads(result))
    print("[EVALUATION] Done fetching")
    queue.close()
    return final_result



class GameManager(multiprocessing.Process):
    """
    Used to manage a Queue of process. In charge of the interaction
    between the processes.
    """

    def __init__(self, game_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.game_queue = game_queue
        self.result_queue = result_queue

    def run(self):
        """ Execute a task from the game_queue """

        process_name = self.name
        while True:
            next_task = self.game_queue.get()

            ## End the processes that are done
            if next_task is None:
                self.game_queue.task_done()
                break

            answer = next_task()
            self.game_queue.task_done()
            self.result_queue.put(answer)




class Game:
    """ A single process that is used to play a game between 2 agents """

    def __init__(self, player, id, color="black", mcts_flag=MCTS_FLAG, goban_size=GOBAN_SIZE, opponent=False):
        self.mcts_flag = mcts_flag
        self.goban_size = goban_size
        self.id = id + 1
        self.player = player
        self.opponent = opponent
        self.board = self._create_board(color)
        self.player_color = 2 if color == "black" else 1
    

    def _create_board(self, color):
        """
        Create a board with a goban_size and the color is
        for the starting player
        """
    
        board = Board(color, self.goban_size)
        board.reset()
        return board
    

    def _prepare_state(self, state):
        """
        Transform the numpy state into a PyTorch tensor with cuda
        if available
        """

        x = torch.from_numpy(np.array([state]))
        x = Variable(x).type(DTYPE_FLOAT)
        return x
    

    def _swap_color(self):
        if self.player_color == 1:
            self.player_color = 2
        else:
            self.player_color = 1

    
    def _get_move(self, board, probas):
        """ Select a move without MCTS """

        player_move = None
        valid_move = False
        can_pass = False
        legal_moves = board.get_legal_moves()

        while valid_move is False and can_pass is False:
            if (len(legal_moves) == 1 and \
                legal_moves[0] == self.goban_size ** 2) or len(legal_moves) == 0:
                can_pass = True
                player_move = self.goban_size ** 2

            if player_move is not None: 
                valid_move = board.test_move(player_move)
                if valid_move is False and can_pass is False:
                    legal_moves.remove(player_move)

            while player_move not in legal_moves and len(legal_moves) > 0:
                player_move = np.random.choice(probas.shape[0], p=probas)
                if player_move not in legal_moves:
                    old_proba = probas[player_move]
                    probas = probas + (old_proba / (probas.shape[0] - 1))
                    probas[player_move] = 0

        return player_move


    def _play(self, state, player):
        """ Choose a move depending on MCTS or not """

        # if self.mcts_flag:
        #     action_scores = player.mcts.search()
        # else:
        feature_maps = player.extractor(state)
        probas = player.policy_net(feature_maps)[0] \
                            .cpu().data.numpy()
        if player.passed is True:
            player_move = self.goban_size ** 2
        else:
            player_move = self._get_move(self.board, probas)

        if player_move == self.goban_size ** 2:
            player.passed = True

        state, reward, done = self.board.step(player_move)
        return state, reward, done, player_move


    def __call__(self):
        """
        Make a game between the player and the opponent and return all the states
        and the associated move. Also returns the winner in order to create the
        training dataset
        """

        done = False
        state = self.board.reset()
        dataset = []
        moves = 0

        while not done:

            ## Prevent cycling in 2 atari situations
            if moves > 60 * self.goban_size:
                return False

            ## For evaluation
            if self.opponent:
                state, reward, done, _ = self._play(self._prepare_state(state), self.player)
                state, reward, done, _ = self._play(self._prepare_state(state), self.opponent)
                moves += 2
            ## For self-play
            else:
                state = self._prepare_state(state)
                new_state, reward, done, player_move = self._play(state, self.player)
                self._swap_color()
                dataset.append((state.cpu().data.numpy(), player_move, \
                                self.player_color))
                state = new_state 
                moves += 1
            
        ## Pickle the result because multiprocessing
        if self.opponent:
            return pickle.dumps([reward])
        return pickle.dumps((dataset, reward))

    
    def solo_play(self, move=None):
        """ Used to play against a human or for GTP, cant be called
        in a multiprocess scenario """

        ## Agent plays the first move of the game
        if move is None:
            state = self._prepare_state(self.board.state)
            state, reward, done, player_move = self._play(state, self.player)
            self._swap_color()
            return player_move
        ## Otherwise just play a move and answer it
        else:
            state, reward, done = self.board.step(move)
            self._swap_color()
            return True
    

    def reset(self):
        state = self.board.reset()

