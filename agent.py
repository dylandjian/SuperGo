from board.gym import make
import numpy as np

## Setup env for 9x9 Go board
env = make('Go9x9-v0')
env.reset()


## Number of last states to keep
HISTORY = 7
GOBAN_SIZE = 9


## BLACK = 0 WHITE = 1


def state_history(history, new_state, color):
    history = np.roll(history, 1, axis=1)
    history[color][-1] = np.array(new_state)
    return history


def create_state(history, board):
    print(board)


done = False
## Contains black and white history, dimensions of this array is 2 x HISTORY x GOBAN_SIZE
history = np.zeros((2, HISTORY, GOBAN_SIZE ** 2))


while done is False:
    print("la valeur du joueur est : %d\n" % (int(env.player_color) - 1))
    env.render()
    break
