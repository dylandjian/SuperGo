import numpy as np
from go import GoEnv as Board
from torch.autograd import Variable
import torch
from const import *
import click
import timeit
from models import agent, feature


def debug(board, state, reward, done):
    board.render()
    print("\nreward value: %d" % reward)
    print("done: %s\n" % done)


def create_board(color="black"):
    """ Create a board with a GOBAN_SIZE from the const file and the color is
        for the starting player """
 
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


def create_dataset(player, opponent, feature_extractor, board):
    for i in range(SELF_PLAY_MATCH):

        board.reset()
        done = False
        state = board.state

        while not done:
            x = prepare_state(state)
            feature_maps = feature_extractor(x)

            move = player.look_ahead(feature_maps)
            state, reward, done = board.step(move)
            # debug(board, state, reward, done)

            # move = player.look_ahead(feature_maps)
            # print(move, type(move))
            # state, reward, done = board.step(move)



@click.command()
@click.option("--human/--no_human", default=False, help="Whether to play against it or not")
def main(human):

    board = create_board(color="white")
    done = False

    inplanes = (HISTORY + 1) * 2 + 1
    ## Probabilities for all moves + pass
    outplanes = (GOBAN_SIZE ** 2) + 1

    ## Init the 2 players
    player = agent.SuperAgent(OUTPLANES_MAP, outplanes)
    opponent = agent.SuperAgent(OUTPLANES_MAP, outplanes)
    feature_extractor = feature.FeatureExtractor(inplanes, OUTPLANES_MAP)

    dataset = create_dataset(player, opponent, feature_extractor, board)
        


if __name__ == "__main__":
    main()