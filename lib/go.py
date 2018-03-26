import pachi_py
import numpy as np
import sys
import six
from const import HISTORY, GOBAN_SIZE



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
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
        """
        self.board_size = board_size
        self.history = [np.zeros((HISTORY + 1, board_size, board_size)),
                        np.zeros((HISTORY + 1, board_size, board_size))]

        colormap = {
            'black': pachi_py.BLACK,
            'white': pachi_py.WHITE,
        }
        self.player_color = colormap[player_color]

        # Filled in by _reset()
        self.state = _format_state(self.history,
                        self.player_color, self.board_size)
        self.done = True


    def get_legal_moves(self):
        return self.board.get_legal_coords(self.player_color)


    def _act(self, action, history):
        """
        Executes an action for the current player
        """

        self.board = self.board.play(action, self.player_color)
        board = self.board.encode()
        color = self.player_color - 1
        history[color] = np.roll(history[color], 1, axis=0)
        history[color][0] = np.array(board[color])
        self.player_color = pachi_py.stone_other(self.player_color)


    def reset(self):
        self.board = pachi_py.CreateBoard(self.board_size)

        opponent_resigned = False
        self.done = self.board.is_terminal or opponent_resigned

        return _format_state(self.history, self.player_color, self.board_size)

    def render(self):
        outfile = sys.stdout
        outfile.write('To play: {}\n{}\n'.format(six.u(
                        pachi_py.color_to_str(self.player_color)),
                        self.board.__repr__().decode()))
        return outfile

    def step(self, action):
        # If already terminal, then don't do anything
        if self.done:
            return _format_state(self.history, self.player_color, self.board_size), \
                     0., True

        # Play
        try:
            self._act(action, self.history)
        except pachi_py.IllegalMove:
            six.reraise(*sys.exc_info())

        # Reward: if nonterminal, then the reward is 0
        if not self.board.is_terminal:
            self.done = False
            return _format_state(self.history, self.player_color, self.board_size), \
                    0., False

        # We're in a terminal state. Reward is 1 if won, -1 if lost
        assert self.board.is_terminal
        self.done = True
        white_wins = self.board.official_score > 0
        black_wins = self.board.official_score < 0
        player_wins = (white_wins and self.player_color == pachi_py.WHITE) or (black_wins and self.player_color == pachi_py.BLACK)
        reward = 1. if player_wins else -1. if (white_wins or black_wins) else 0.
        return _format_state(self.history, self.color), rewa, self.board_sizerd, True
