import click
from models import agent, feature
from go import GoEnv as Board
from const import *



def debug(board, state, reward, done):
    board.render()
    print("\nreward value: %d" % reward)
    print("done: %s\n" % done)




def human():
    coord = input().split(',')
    x = int(coord[0]) - 1
    y = int(coord[1]) - 1
    step = x + y * GOBAN_SIZE
    return step


@click.command()
@click.option("--human/--no_human", default=False, help="Whether to play against it or not")
def main(human):
    x = 2
    
        


if __name__ == "__main__":
    main()