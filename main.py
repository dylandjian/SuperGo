import numpy as np
from go import GoEnv as Board
from torch.autograd import Variable
import torch
from const import *
import click
import timeit
from models import agent, feature
import threading
from queue import Queue



class GameThread(threading.Thread):
    """ A single thread that is used to play a game """

    def __init__(self, threadID, name, player, opponent, extractor):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.extractor = extractor
        self.board = create_board()
        self.player = player
        self.opponent = opponent
    
    def run(self):
        """
        Make a game between the player and the opponent and return all the states
        and the associated probabilities of each move. Also returns the winner in
        order to create the training dataset
        """

        done = False
        state = self.board.reset()

        while not done:
            x = prepare_state(state)
            feature_maps = self.extractor(x)

            move = self.player.look_ahead(feature_maps)
            state, reward, done = self.board.step(move)
            return x
            # debug(board, state, reward, done)

            # move = player.look_ahead(feature_maps)
            # print(move, type(move))
            # state, reward, done = board.step(move)


def debug(board, state, reward, done):
    board.render()
    print("\nreward value: %d" % reward)
    print("done: %s\n" % done)


def create_board(color="black"):
    """
    Create a board with a GOBAN_SIZE from the const file and the color is
    for the starting player
    """
 
    board = Board(color, GOBAN_SIZE)
    return board


def human():
    coord = input().split(',')
    x = int(coord[0]) - 1
    y = int(coord[1]) - 1
    step = x + y * GOBAN_SIZE
    return step


def prepare_state(state):
    """
    Transform the numpy state into a PyTorch tensor with cuda
    if available
    """

    x = torch.from_numpy(np.array([state]))
    if CUDA:
        x = Variable(x).type(torch.FloatTensor).cuda()
    else:
        x = Variable(x).type(torch.FloatTensor)
    return x


def create_dataset(player, opponent, feature_extractor):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    game_threads = Queue()
    for id in range(THREADS):
        game = GameThread(id, "Game {}".format(id), \
                        player, opponent, feature_extractor)
        game.start()

    for i in range(SELF_PLAY_MATCH // THREADS):
        game_threads.join()


@click.command()
@click.option("--human/--no_human", default=False, help="Whether to play against it or not")
def main(human):

    inplanes = (HISTORY + 1) * 2 + 1
    ## Probabilities for all moves + pass
    outplanes = (GOBAN_SIZE ** 2) + 1

    ## Init the 2 players
    player = agent.SuperAgent(OUTPLANES_MAP, outplanes)
    opponent = agent.SuperAgent(OUTPLANES_MAP, outplanes)

    if CUDA:
        extractor = feature.Extractor(inplanes, OUTPLANES_MAP).cuda()
    else:
        extractor = feature.Extractor(inplanes, OUTPLANES_MAP)
    
    x = start_time = timeit.default_timer()
    dataset = create_dataset(player, opponent, extractor)
    print('Total time for create dataset: %.5fs' % x)
        


if __name__ == "__main__":
    main()