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
    

    def create_board(self, color="black"):
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

    def __init__(self, player, opponent, extractor):
        self.player = player
        self.opponent = opponent
        self.extractor = extractor
    

    def _prepare_state(self, state):
        """
        Transform the numpy state into a PyTorch tensor with cuda
        if available
        """

        x = torch.from_numpy(np.array([state]))
        x = Variable(x).type(DTYPE)
        return x
    

    def _draw_move(self, action_scores, competitive=False):
        """
        Find the best move, either deterministically for competitive play
        or stochiasticly according to some temperature constant
        """

        if competitive:
            move = np.argmax(action_scores)

        else:
            action_scores = np.power(action_scores, (1. / TEMP))
            total = np.sum(action_scores)
            probas = action_scores / total
            move = np.random.choice(action_scores.shape[0], p=probas)

        return move


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
            x = self._prepare_state(state)
            feature_maps = self.extractor(x)

             = MCTS(state, C_PUCT)
            move = self._draw_move(counts)
            state, reward, done = board.step(move)
            dataset.append([x, move])

            ## Here we shape the training dataset with a state, 
            ## the output of the MCTS and a placeholder for the
            ## winner (either 1 or -1)
            break
    
        ## Pickle the result because multiprocessing
        return pickle.dumps([dataset, reward])
