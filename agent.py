from board.gym import make
import numpy as np
from const import *



## Setup env for 9x9 Go board
env = make('Go9x9-v0')
env.reset()


def state_history(history, new_state, color):
    history = np.roll(history, 1, axis=1)
    history[color][-1] = np.array(new_state)
    return history


def create_state(history, color, board):
    last_layer = np.full(GOBAN_SIZE ** 2, color)
    print(board)


done = False
## Contains black and white history, dimensions of this array is 2 x HISTORY x GOBAN_SIZE
history = np.zeros((2, HISTORY, GOBAN_SIZE ** 2))


for epoch in range(1, EPOCHS):
    while done is False:
        print("la valeur du joueur est : %d\n" % (int(env.player_color) - 1))
        env.render()
        break
