import pachi_py
import six


def _pass_action(board_size):
    return board_size ** 2

def _resign_action(board_size):
    return board_size ** 2 + 1

def _coord_to_action(board, c):
    '''Converts Pachi coordinates to actions'''
    if c == pachi_py.PASS_COORD: return _pass_action(board.size)
    if c == pachi_py.RESIGN_COORD: return _resign_action(board.size)
    i, j = board.coord_to_ij(c)
    return i*board.size + j

def _action_to_coord(board, a):
    '''Converts actions to Pachi coordinates'''
    if a == _pass_action(board.size): return pachi_py.PASS_COORD
    if a == _resign_action(board.size): return pachi_py.RESIGN_COORD
    return board.ij_to_coord(a // board.size, a % board.size)

def str_to_action(board, s):
    return _coord_to_action(board, board.str_to_coord(s.encode()))


class GoState():
    """
    Go game state. Consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is different
    from Pachi's internal "coord_t" encoding.
    """

    def __init__(self, board, color):
        self.board, self.color = board, color

    def act(self, action):
        """
        Executes an action for the current player

        Returns:
            a new GoState with the new board and the player switched
        """

        return GoState(
            self.board.play(_action_to_coord(self.board, action), self.color),
            pachi_py.stone_other(self.color))

    def __repr__(self):
        return 'To play: {}\n{}'.format(six.u(pachi_py.color_to_str(self.color)),
                                        self.board.__repr__().decode())



class GoEnv():

    def __init__(self, player_color, opponent, observation_type, illegal_move_mode, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            illegal_move_mode: What to do when the agent makes an illegal move. Choices: 'raise' or 'lose'
        """
        self.board_size = board_size

        self._seed()

        colormap = {
            'black': pachi_py.BLACK,
            'white': pachi_py.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error("player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent_policy = None
        self.opponent = opponent

        assert observation_type in ['image3c']
        self.observation_type = observation_type

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        if self.observation_type != 'image3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        shape = pachi_py.CreateBoard(self.board_size).encode().shape
        self.observation_space = spaces.Box(np.zeros(shape), np.ones(shape))
        # One action for each board position, pass, and resign
        self.action_space = spaces.Discrete(self.board_size**2 + 2)

        # Filled in by _reset()
        self.state = None
        self.done = True

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        pachi_py.pachi_srand(seed2)
        return [seed1, seed2]

    def _reset(self):
        self.state = GoState(pachi_py.CreateBoard(self.board_size), pachi_py.BLACK)

        # (re-initialize) the opponent
        # necessary because a pachi engine is attached to a game via internal data in a board
        # so with a fresh game, we need a fresh engine
        # self._reset_opponent(self.state.board)

        # Let the opponent play if it's not the agent's turn
        opponent_resigned = False
        # if self.state.color != self.player_color:
        #     self.state, opponent_resigned = self._exec_opponent_play(self.state, None, None)

        # We should be back to the agent color
        # assert self.state.color == self.player_color

        self.done = self.state.board.is_terminal or opponent_resigned
        return self.state.board.encode()

    def _close(self):
        self.opponent_policy = None
        self.state = None

    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.state) + '\n')
        return outfile

    def _step(self, action):
        # assert self.state.color == self.player_color

        # If already terminal, then don't do anything
        if self.done:
            return self.state.board.encode(), 0., True, {'state': self.state}

        # If resigned, then we're done
        if action == _resign_action(self.board_size):
            self.done = True
            return self.state.board.encode(), -1., True, {'state': self.state}

        # Play
        prev_state = self.state
        try:
            self.state = self.state.act(action)
        except pachi_py.IllegalMove:
            if self.illegal_move_mode == 'raise':
                six.reraise(*sys.exc_info())
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state.board.encode(), -1., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal move action: {}'.format(self.illegal_move_mode))

        # Opponent play
        if not self.state.board.is_terminal:
            self.state, opponent_resigned = self._exec_opponent_play(self.state, prev_state, action)
            # After opponent play, we should be back to the original color
            assert self.state.color == self.player_color

            # If the opponent resigns, then the agent wins
            if opponent_resigned:
                self.done = True
                return self.state.board.encode(), 1., True, {'state': self.state}

        # Reward: if nonterminal, then the reward is 0
        if not self.state.board.is_terminal:
            self.done = False
            return self.state.board.encode(), 0., False, {'state': self.state}

        # We're in a terminal state. Reward is 1 if won, -1 if lost
        assert self.state.board.is_terminal
        self.done = True
        white_wins = self.state.board.official_score > 0
        black_wins = self.state.board.official_score < 0
        player_wins = (white_wins and self.player_color == pachi_py.WHITE) or (black_wins and self.player_color == pachi_py.BLACK)
        reward = 1. if player_wins else -1. if (white_wins or black_wins) else 0.
        return self.state.board.encode(), reward, True, {'state': self.state}
