import numpy as np
import time
import pickle
from go import GoEnv as Board
from torch.autograd import Variable
import torch
from const import *
import click
import timeit
from models import agent, feature
import multiprocessing



class GameManager(multiprocessing.Process):
    def __init__(self, game_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.game_queue = game_queue
        self.result_queue = result_queue
    
    def run(self):
        """
        Execute a task from the game_queue
        """

        process_name = self.name
        while True:
            next_task = self.game_queue.get()
            if next_task is None:
                print('{} is done'.format(process_name))
                self.game_queue.task_done()
                break

            print('Starting new task on {}'.format(process_name))
            answer = next_task()
            self.game_queue.task_done()
            self.result_queue.put(answer)




class Game:
    """ A single process that is used to play a game """

    def __init__(self, player, opponent, extractor):
        self.board = create_board()
        self.player = player
        self.opponent = opponent
        self.extractor = extractor
    
    def _prepare_state(self, state):
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


    def __call__(self):
        """
        Make a game between the player and the opponent and return all the states
        and the associated probabilities of each move. Also returns the winner in
        order to create the training dataset
        """

        done = False
        state = self.board.reset()

        # while not done:
        x = self._prepare_state(state)
            # feature_maps = self.extractor(x)

            # move = self.player.look_ahead(feature_maps)
            # state, reward, done = self.board.step(move)
        return pickle.dumps(x)
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




def create_dataset(player, opponent, extractor):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    queue = multiprocessing.JoinableQueue()
    dataset = multiprocessing.Queue()
    train_dataset = []

    game_managers = [
        GameManager(queue, dataset)
        for _ in range(CPU_CORES)
    ]

    for game_manager in game_managers:
        game_manager.start()

    for _ in range(NUM_MATCHES):
        queue.put(Game(player, opponent, extractor))
    
    for _ in range(CPU_CORES):
        queue.put(None)
    
    queue.join()
    
    for _ in range(NUM_MATCHES):
        result = dataset.get()
        train_dataset.append(pickle.loads(result))
    
    return train_dataset
        


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
    
    start = time.clock()
    print('Starting\n')
    dataset = create_dataset(player, opponent, extractor)
    end = time.clock()
    print(end - start)
        


if __name__ == "__main__":
    main()