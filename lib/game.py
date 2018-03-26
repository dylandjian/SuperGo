import multiprocessing
import numpy as np
from .go import GoEnv as Board
import pickle
from const import *
from torch.autograd import Variable
from .mcts import MCTS


class GameManager(multiprocessing.Process):
    """
    Used to manage a Queue of process. In charge of the interaction
    between the processes.
    """

    def __init__(self, game_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.board = self.create_board()
        self.game_queue = game_queue
        self.result_queue = result_queue
    

    def create_board(self, color="white"):
        """
        Create a board with a GOBAN_SIZE from the const file and the color is
        for the starting player
        """
    
        board = Board(color, GOBAN_SIZE)
        return board


    def run(self):
        """ Execute a task from the game_queue """

        process_name = self.name
        while True:
            next_task = self.game_queue.get()

            ## End the processes that are done
            if next_task is None:
                print('{} is done'.format(process_name))
                self.game_queue.task_done()
                break

            print('Starting new task on {}'.format(process_name))
            answer = next_task(self.board)
            self.game_queue.task_done()
            self.result_queue.put(answer)




class Game:
    """ A single process that is used to play a game between 2 agents """

    def __init__(self, player, opponent):
        self.players = [player, opponent]
    

    def _prepare_state(self, state):
        """
        Transform the numpy state into a PyTorch tensor with cuda
        if available
        """

        x = torch.from_numpy(np.array([state]))
        x = Variable(x).type(DTYPE)
        return x
    

    def __call__(self, board):
        """
        Make a game between the player and the opponent and return all the states
        and the associated move. Also returns the winner in order to create the
        training dataset
        """

        done = False
        state = board.reset()
        dataset = []

        while not done:
            for player in self.players:
                x = self._prepare_state(state)
                feature_maps = player.extractor(x)

                player_move = player.policy_net(feature_maps)
                print(player_move)
                assert 0
                state, reward, done = board.step(player_move)
                dataset.append([x, player_move, 1])

            break
            ## Here we shape the training dataset with a state, 
            ## the output of the MCTS and a placeholder for the
            ## winner (either 1 or -1)
    
        ## Pickle the result because multiprocessing
        return pickle.dumps([dataset, reward])
