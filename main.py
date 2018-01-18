from board.gym import make
import numpy as np
from const import *
import click
import timeit



def create_board():
    """ Create a board with the modified gym/go package """
 
    board = make('Go9x9-v0')
    board.reset()
    return board


def asyncopt(result):
    boards.append(result)
    return boards



@click.command()
@click.option("--human/--no_human", default=False, help="Whether to play against it or not")
def main(human):

    board = create_board()

    # while not board.done:
    #     board.render()
    #     if human:
    #         coord = input().split(',')
    #         x = int(coord[0]) - 1
    #         y = int(coord[1]) - 1
    #         step = x + y * GOBAN_SIZE
    #         board.step(step)

    state, reward, done, _ = board.step(-1)
    # state, reward, done, _ = board.step(83)
    board.render()

    print(state, reward, done)





if __name__ == "__main__":
    main()