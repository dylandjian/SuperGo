from torch.utils.data import Dataset, DataLoader
from const import *
import numpy as np
import timeit

class SelfPlayDataset(Dataset):
    """
    Self-play dataset containing state, probabilities
    and the winner of the game.
    """

    def __init__(self):
        """ Instanciate a dataset """

        self.states = np.zeros((MOVES, (HISTORY + 1) * 2 + 1, GOBAN_SIZE, GOBAN_SIZE))
        if NO_MCTS:
            self.plays = np.zeros((MOVES, GOBAN_SIZE ** 2 + 1))
        else:
            self.plays = np.zeros(MOVES)
        self.winners = np.zeros(MOVES)
        self.current_len = 0

    def __len__(self):
        return self.current_len

    def __getitem__(self, idx):
        return self.states[idx], self.plays[idx], \
               self.winners[idx]

    def update(self, game):
        dataset = np.array(game[0])
        number_moves = dataset.shape[0]
        self.current_len = min(self.current_len + number_moves, MOVES)
        
        self.states = np.roll(self.states, number_moves, axis=0)
        self.states[:number_moves] = np.vstack(dataset[:,0])

        self.plays = np.roll(self.plays, number_moves, axis=0)
        if NO_MCTS:
            self.plays[np.arange(number_moves),np.hstack(dataset[:,1])] = 1
        else:
            self.plays[:number_moves] = np.hstack(dataset[:,1])

        self.winners = np.roll(self.winners, number_moves, axis=0)
        winners = dataset[:,2]
        winners[np.where(winners - 1 != game[1])] = -1
        winners[np.where(winners != -1)] = 1
        self.winners[:number_moves] = winners
    
    def update_batch(self, raw_dataset):
        for game in raw_dataset:
            self.update(game)
