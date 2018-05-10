import numpy as np
import pickle
from const import *
from models.mcts import MCTS
from .go import GoEnv as Board
from .utils import _prepare_state


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
        legal_moves = board.get_legal_moves()

        while player_move not in legal_moves and len(legal_moves) > 0:
            player_move = np.random.choice(probas.shape[0], p=probas)
            if player_move not in legal_moves:
                old_proba = probas[player_move]
                probas = probas + (old_proba / (probas.shape[0] - 1))
                probas[player_move] = 0

        return player_move


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
            _, probas = player.predict(state)
            probas = probas[0].cpu().data.numpy()
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

            ## Prevent game from cycling
            if moves > MOVE_LIMIT:
                reward = self.board.get_winner()
                if self.opponent:
                    print("[EVALUATION] Match %d done in eval after max move, winner %s"
                        % (self.id, "black" if reward == 0 else "white"))
                    return pickle.dumps([reward])
                return pickle.dumps((dataset, reward)) 
            
            ## Adaptative temperature to stop exploration
            if moves > TEMPERATURE_MOVE:
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
            print("[EVALUATION] Match %d done in eval after %d moves, winner %s" % (self.id,
                        moves, "black" if reward == 0 else "white"))
            return pickle.dumps([reward])

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
