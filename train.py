from lib import game, mcts
import multiprocessing
from models import feature, value, policy
from const import *
import pickle
import click

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


class Player:

    def __init__(self):
        if CUDA:
            self.extractor = feature.Extractor(INPLANES, OUTPLANES_MAP).cuda()
            self.value_net = value.ValueNet(OUTPLANES_MAP).cuda()
            self.policy_net = policy.PolicyNet(OUTPLANES_MAP, OUTPLANES).cuda()
        else:
            self.extractor = feature.Extractor(INPLANES, OUTPLANES_MAP)
            self.value_net = value.ValueNet(OUTPLANES_MAP)
            self.policy_net = policy.PolicyNet(OUTPLANES_MAP, OUTPLANES)    
        self.mcts = mcts.MCTS(C_PUCT, self.extractor, self.value_net, self.policy_net)


def create_dataset(player, opponent):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    queue = multiprocessing.JoinableQueue()
    dataset = multiprocessing.Queue()
    train_dataset = []

    game_managers = [
        game.GameManager(queue, dataset)
        for _ in range(CPU_CORES)
    ]

    for game_manager in game_managers:
        game_manager.start()

    for _ in range(SELF_PLAY_MATCH):
        queue.put(game.Game(player, opponent))
    
    for _ in range(CPU_CORES):
        queue.put(None)
    
    queue.join()
    
    for _ in range(SELF_PLAY_MATCH):
        result = dataset.get()
        train_dataset.append(pickle.loads(result))
    
    print(train_dataset)
    return train_dataset


@click.command()
def main():
    player = Player()
    dataset = create_dataset(player, player)
    print(dataset)


if __name__ == "__main__":
    main()