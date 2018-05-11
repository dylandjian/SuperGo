import pachi_py
from copy import deepcopy
import numpy as np
import sys
import six
from const import HISTORY, GOBAN_SIZE


def _pass_action(board_size):
    return board_size ** 2


def _resign_action(board_size):
    return board_size ** 2 + 1


def _coord_to_action(board, c):
    """ Converts Pachi coordinates to actions """

    if c == pachi_py.PASS_COORD:
        return _pass_action(board.size)
    if c == pachi_py.RESIGN_COORD:
        return _resign_action(board.size)

    i, j = board.coord_to_ij(c)
    return i*board.size + j


def _action_to_coord(board, a):
    """ Converts actions to Pachi coordinates """

    if a == _pass_action(board.size):
        return pachi_py.PASS_COORD
    if a == _resign_action(board.size):
        return pachi_py.RESIGN_COORD

    return board.ij_to_coord(a // board.size, a % board.size)


def _format_state(history, player_color, board_size):
    """ 
    Format the encoded board into the state that is the input
    of the feature model, defined in the AlphaGo Zero paper 
    BLACK = 1
    WHITE = 2
    """

    state_history = np.concatenate((history[0], history[1]), axis=0)
    to_play = np.full((1, board_size, board_size), player_color - 1)
    final_state = np.concatenate((state_history, to_play), axis=0)
    return final_state
    


class GoEnv():

    def __init__(self, player_color, board_size):
        self.board_size = board_size
        self.history = [np.zeros((HISTORY + 1, board_size, board_size)),
                        np.zeros((HISTORY + 1, board_size, board_size))]

        colormap = {
            'black': pachi_py.BLACK,
            'white': pachi_py.WHITE,
        }
        self.player_color = colormap[player_color]

        self.komi = self._get_komi(board_size)
        self.state = _format_state(self.history,
                        self.player_color, self.board_size)
        self.done = True


    def _get_komi(self, board_size):
        """ Initialize a komi depending on the size of the board """

        if 14 <= board_size <= 19:
            return 7.5
        elif 9 <= board_size <= 13:
            return 5.5
        return 0
    

    def get_legal_moves(self):
        """ Get all the legal moves and transform their coords into 1d """

        legal_moves = self.board.get_legal_coords(self.player_color, filter_suicides=True)
        final_moves = []

        for pachi_move in legal_moves:
            move = _coord_to_action(self.board, pachi_move)
            if move != self.board_size ** 2 or self.test_move(move):
                final_moves.append(move)
        
        if len(final_moves) == 0:
            return [self.board_size ** 2]

        return final_moves


    def _act(self, action, history):
        """ Executes an action for the current player """

        self.board = self.board.play(_action_to_coord(self.board, action), self.player_color)
        board = self.board.encode()
        color = self.player_color - 1
        history[color] = np.roll(history[color], 1, axis=0)
        history[color][0] = np.array(board[color])
        self.player_color = pachi_py.stone_other(self.player_color)


    def test_move(self, action):
        """
        Test if a specific valid action should be played,
        depending on the current score. This is used to stop
        the agent from passing if it makes him loose
        """

        board_clone = self.board.clone()
        current_score = board_clone.fast_score  + self.komi

        board_clone = board_clone.play(_action_to_coord(board_clone, action), self.player_color)
        new_score = board_clone.fast_score + self.komi

        if self.player_color - 1 == 0 and new_score >= current_score \
           or self.player_color - 1 == 1 and new_score <= current_score:
           return False
        return True


    def reset(self):
        """ Reset the board """

        self.board = pachi_py.CreateBoard(self.board_size)
        opponent_resigned = False
        self.done = self.board.is_terminal or opponent_resigned
        return _format_state(self.history, self.player_color, self.board_size)


    def render(self):
        """ Print the board for human reading """

        outfile = sys.stdout
        outfile.write('To play: {}\n{}\n'.format(six.u(
                        pachi_py.color_to_str(self.player_color)),
                        self.board.__repr__().decode()))
        return outfile


    def get_winner(self):
        """ Get the winner, using the Tromp Taylor scoring + the komi """

        score = self.board.fast_score + self.komi
        white_wins = self.board.fast_score > 0
        black_wins = self.board.fast_score < 0
        reward = 1 if white_wins else 0

        return reward
    

    def step(self, action):
        """ Perfoms an action and choose the winner if the 2 player
            have passed """

        if not self.done:
            try:
                self._act(action, self.history)
            except pachi_py.IllegalMove:
                six.reraise(*sys.exc_info())

        # Reward: if nonterminal, then the reward is -1
        if not self.board.is_terminal:
            return _format_state(self.history, self.player_color, self.board_size), \
                    -1, False

        assert self.board.is_terminal
        self.done = True
        reward = self.get_winner()
        return _format_state(self.history, self.player_color, self.board_size), reward, True


    def __deepcopy__(self, memo):
        """ Used to overwrite the deepcopy implicit method since
            the board cannot be deepcopied """

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "board":
                setattr(result, k, self.board.clone())
            else:
                setattr(result, k, deepcopy(v, memo))
        return result
