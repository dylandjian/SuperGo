from lib import train, play
from lib.dataset import SelfPlayDataset
from lib.evaluate import evaluate
from models.agent import Player
from const import *
import pickle
import timeit
import click
from torch.utils.data import DataLoader


from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass



@click.command()
def main():
    player = Player()
    dataset = SelfPlayDataset() 
    start_time = timeit.default_timer()
    dataset.update(play.create_matches(player))
    total_time = timeit.default_timer() - start_time
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1)
    print("DATASET LENGTH: ", len(dataloader))
    print('Dataset ready in: ---- %.3f seconds for %d self-play games ----' % (
        total_time, SELF_PLAY_MATCH))
    print('Average time per game: ---- %.3f seconds with %d processes ----' %
             (total_time / SELF_PLAY_MATCH, CPU_CORES))
    
    new_player = train.train(dataloader, player)
    print(id(new_player))
    print(id(player))
    assert 0
    if evaluate(player, new_player):
        player = new_player


if __name__ == "__main__":
    main()