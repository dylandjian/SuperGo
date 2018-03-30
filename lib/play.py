import multiprocessing
import random
import numpy as np
from .go import GoEnv as Board
import pickle
from const import *
from torch.autograd import Variable



def create_matches(player, opponent=None, match_number=SELF_PLAY_MATCH):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    queue = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    game_results = []

    game_managers = [
        GameManager(queue, results)
        for _ in range(CPU_CORES)
    ]

    for game_manager in game_managers:
        game_manager.start()

    for id in range(match_number):
        queue.put(Game(player, id, opponent=opponent))
    
    for _ in range(CPU_CORES):
        queue.put(None)
    
    queue.join()
    
    print("--- Starting to fetch results ---")
    for _ in range(match_number):
        result = results.get()
        if result:
            game_results.append(pickle.loads(result))
    print("--- Done fetching ---")
    queue.close()
    return game_results




class GameManager(multiprocessing.Process):
    """
    Used to manage a Queue of process. In charge of the interaction
    between the processes.
    """

    def __init__(self, game_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.game_queue = game_queue
        self.result_queue = result_queue


    def run(self):
        """ Execute a task from the game_queue """

        process_name = self.name
        while True:
            next_task = self.game_queue.get()

            ## End the processes that are done
            if next_task is None:
                self.game_queue.task_done()
                break

            answer = next_task()
            self.game_queue.task_done()
            self.result_queue.put(answer)




class Game:
    """ A single process that is used to play a game between 2 agents """

    def __init__(self, player, id, opponent=False):
        self.board = self._create_board()
        self.id = id
        self.player = player
        self.opponent = opponent
    

    def _create_board(self, color="white"):
        """
        Create a board with a GOBAN_SIZE from the const file and the color is
        for the starting player
        """
    
        board = Board(color, GOBAN_SIZE)
        board.reset()
        return board
    

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
    
    
    def _get_move(self, board, probas):
        player_move = None
        valid_move = False
        can_pass = False
        legal_moves = board.get_legal_moves()

        while valid_move is False and can_pass is False:
            if (len(legal_moves) == 1 and \
                legal_moves[0] == GOBAN_SIZE ** 2) or len(legal_moves) == 0:
                can_pass = True
                player_move = GOBAN_SIZE ** 2

            if player_move is not None: 
                valid_move = board.test_move(player_move)
                if valid_move is False and can_pass is False:
                    legal_moves.remove(player_move)

            while player_move not in legal_moves and len(legal_moves) > 0:
                player_move = np.random.choice(probas.shape[0], p=probas)

        return player_move


    def _play(self, state, player):

        feature_maps = player.extractor(state)
        probas = player.policy_net(feature_maps)\
                            .cpu().data.numpy()

        if player.passed is True:
            player_move = GOBAN_SIZE ** 2
        else:
            player_move = self._get_move(self.board, probas)

        if player_move == GOBAN_SIZE ** 2:
            player.passed = True

        state, reward, done = self.board.step(player_move)
        return state, reward, done, player_move


    def __call__(self):
        """
        Make a game between the player and the opponent and return all the states
        and the associated move. Also returns the winner in order to create the
        training dataset
        """

        print("[%d] Starting" % (self.id + 1))

        done = False
        state = self.board.reset()
        dataset = []
        moves = 0


        while not done:
            ## Prevent cycling in 2 atari situations
            ## poor fix, to improve
            if moves > 60 * GOBAN_SIZE:
                return False

            if self.opponent:
                self.board.render()
                state, reward, done, _ = self._play(self._prepare_state(state), self.player)
                state, reward, done, _ = self._play(self._prepare_state(state), self.opponent)
                moves += 2
            else:
                state = self._prepare_state(state)
                new_state, reward, done, player_move = self._play(state, self.player)
                dataset.append([state.cpu().data.numpy(), player_move, \
                                self.board.player_color])
                state = new_state 
                moves += 1

        print("[%d] Done" % (self.id + 1))

        if self.opponent:
            return pickle.dumps([reward])
        ## Pickle the result because multiprocessing
        return pickle.dumps([dataset, reward])
