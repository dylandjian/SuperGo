from lib import game
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
        game.GameManager(queue, dataset)
        for _ in range(CPU_CORES)
    ]

    for game_manager in game_managers:
        game_manager.start()

    for _ in range(SELF_PLAY_MATCH):
        queue.put(game.Game(player, opponent, extractor))
    
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

    ## Init the 2 players
    if CUDA:
        extractor = feature.Extractor(INPLANES, OUTPLANES_MAP).cuda()
        value_net = value.ValueNet(OUTPLANES_MAP).cuda()
        policy_net = policy.PolicyNet(OUTPLANES_MAP, OUTPLANES).cuda()
    else:
        extractor = feature.Extractor(INPLANES, OUTPLANES_MAP)
        value_net = value.ValueNet(OUTPLANES_MAP)
        policy_net = policy.PolicyNet(OUTPLANES_MAP, OUTPLANES)    
    
    player = [value_net, policy_net]
    dataset = create_dataset(player, player, extractor)


if __name__ == "__main__":
    main()