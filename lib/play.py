import multiprocessing
import timeit
import random
import numpy as np
import pickle
import time
import os
from copy import deepcopy
from models.agent import Player
from models.mcts import MCTS
from .go import GoEnv as Board
from const import *
from pymongo import MongoClient
from torch.autograd import Variable
from .utils import get_player, load_player, _prepare_state


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



def self_play(current_time, loaded_version):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    client = MongoClient()
    collection = client.superGo[current_time]
    game_id = 0
    current_version = 1
    player = False

    while True:

        ## Load the player when restarting traning
        if loaded_version:
            new_player, checkpoint = load_player(current_time, 
                                                loaded_version)
            game_id = collection.find().count()
            current_version = checkpoint['version'] + 1
            loaded_version = False
        else:
            new_player, checkpoint = get_player(current_time, current_version)
            if new_player:
                current_version = checkpoint['version'] + 1

        print("[PLAY] Current improvement level: %d" % current_version)
        if current_version == 1 and not player and not new_player:
            print("[PLAY] Waiting for first player")
            time.sleep(5)
            continue

        if new_player:
            player = new_player
            print("[PLAY] New player !")

        queue, results = create_matches(player , \
                    cores=PARALLEL_SELF_PLAY, match_number=SELF_PLAY_MATCH) 
        print("[PLAY] Starting to fetch fresh games")
        queue.join()
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
        results.close()


def play(player, opponent):
    """ Game between two players, for evaluation """

    queue, results = create_matches(deepcopy(player), opponent=deepcopy(opponent), \
                cores=PARALLEL_EVAL, match_number=EVAL_MATCHS) 
    try:
        queue.join()
        
        print("[EVALUATION] Starting to fetch fresh games")
        final_result = []
        for idx in range(EVAL_MATCHS):
            result = results.get()
            if result:
                final_result.append(pickle.loads(result))
        print("[EVALUATION] Done fetching")
    finally:
        queue.close()
        results.close()
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
            try:
                next_task = self.game_queue.get(600000)

                ## End the processes that are done
                if next_task is None:
                    self.game_queue.task_done()
                    break

                answer = next_task()
                self.game_queue.task_done()
                self.result_queue.put(answer)
            except Exception as e:
                print("xd")




class Game:
    """ A single process that is used to play a game between 2 agents """

    def __init__(self, player, id, color="black", mcts_flag=MCTS_FLAG, goban_size=GOBAN_SIZE, opponent=False):
        self.goban_size = goban_size
        self.id = id + 1
        self.human_pass = False
        self.board = self._create_board(color)
        self.player_color = 2 if color == "black" else 1
        self.mcts = mcts_flag
        if mcts_flag:
            self.mcts = MCTS()
        self.player = player
        self.opponent = opponent

    def _create_board(self, color):
        """
        Create a board with a goban_size and the color is
        for the starting player
        """
    
        board = Board(color, self.goban_size)
        board.reset()
        return board
    

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

        while player_move not in legal_moves and len(legal_moves) > 0:
            player_move = np.random.choice(probas.shape[0], p=probas)
            if player_move not in legal_moves:
                old_proba = probas[player_move]
                probas = probas + (old_proba / (probas.shape[0] - 1))
                probas[player_move] = 0

        return player_move

    # @profile
    def _play(self, state, player, other_pass, competitive=False):
        """ Choose a move depending on MCTS or not """

        if self.mcts:
            if player.passed is True or other_pass:
                action_scores = np.zeros((self.goban_size ** 2 + 1,))
                action_scores[-1] = 1
                action = self.goban_size ** 2
            else:
                action_scores, action = self.mcts.search(self.board, player,\
                                             competitive=competitive)

            if action == self.goban_size ** 2:
                player.passed = True
            
        else:
            feature_maps = player.extractor(state)
            probas = player.policy_net(feature_maps)[0] \
                                .cpu().data.numpy()
            if player.passed is True:
                action = self.goban_size ** 2
            else:
                action = self._get_move(self.board, probas)

            if action == self.goban_size ** 2:
                player.passed = True

            action_scores = np.zeros((self.goban_size ** 2 + 1),)
            action_scores[action] = 1

        state, reward, done = self.board.step(action)
        return state, reward, done, action_scores, action


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
        comp = False

        while not done:
            ## Prevent cycling in 2 atari situations
            if moves > MOVE_LIMIT:
                print("cc")
                return pickle.dumps((dataset, self.board.get_winner()))
            
            if moves > MOVE_LIMIT / 24:
                comp = True

            ## For evaluation
            if self.opponent:
                state, reward, done, _, action = self._play(_prepare_state(state), \
                                                self.player, self.opponent.passed, competitive=True)
                state, reward, done, _, action = self._play(_prepare_state(state), \
                                                self.opponent, self.player.passed, competitive=True)
                moves += 2

            ## For self-play
            else:
                state = _prepare_state(state)
                new_state, reward, done, probas, action = self._play(state, self.player, \
                                                            False, competitive=comp)
                self._swap_color()
                dataset.append((state.cpu().data.numpy(), probas, \
                                self.player_color, action))
                state = new_state 
                moves += 1
            
        ## Pickle the result because multiprocessing
        if self.opponent:
            print("[EVALUATION] Match %d done in eval" % self.id)
            self.opponent.passed = False
            return pickle.dumps([reward])
        self.player.passed = False
        return pickle.dumps((dataset, reward))

    
    def solo_play(self, move=None):
        """ Used to play against a human or for GTP, cant be called
        in a multiprocess scenario """

        ## Agent plays the first move of the game
        if move is None:
            state = _prepare_state(self.board.state)
            state, reward, done, probas, move = self._play(state, self.player, self.human_pass, competitive=True)
            self._swap_color()
            return move
        ## Otherwise just play a move and answer it
        else:
            state, reward, done = self.board.step(move)
            if move != self.board.board_size ** 2:
                self.mcts.advance(move)
            else:
                self.human_pass = True
            self._swap_color()
            return True
    

    def reset(self):
        state = self.board.reset()
    

