from torch.utils.data import Dataset, DataLoader
from const import GOBAN_SIZE
import numpy as np

class SelfPlayDataset(Dataset):
    """
    Self-play dataset containing state, probabilities
    and the winner of the game.
    """

    def __init__(self):
        """ Instanciate a dataset """

        self.states = []
        self.plays = []
        self.winners = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.plays[idx], \
               self.winners[idx]


    def update(self, raw_dataset):
        for game in raw_dataset:
            winner = game[1]
            for data in game[0]:
                ohe = np.zeros(GOBAN_SIZE ** 2 + 1)
                ohe[data[1]] = 1

                self.states.append(data[0][0])
                self.plays.append(ohe)
                self.winners.append(1 if winner == data[2] - 1 else -1)