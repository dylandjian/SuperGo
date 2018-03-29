from torch.utils.data import Dataset, DataLoader

class SelfPlayDataset(Dataset):
    """
    Self-play dataset containing state, probabilities
    and the winner of the game.
    """

    def __init__(self):
        """"""
        self.x = x

    
    def __len__(self):
        return len(self.state_num)


    def __getitem__(self, idx):
        return self.x[idx]
