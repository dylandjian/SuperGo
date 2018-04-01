from lib import train, play
from lib.dataset import SelfPlayDataset
from lib.evaluate import evaluate
from models.agent import Player
from const import *
import pickle
import timeit
import time
import os
import click
from torch.utils.data import DataLoader
from copy import deepcopy


from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass



def save_checkpoint(model, optimizer, current_version, filename, current_time):
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                                'saved_models', str(current_time))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    state = {
        'state_dict': model.state_dict(),
        'version': current_version,
        'optimizer' : optimizer.state_dict(),
    }
    filename = os.path.join(dir_path, "{}-{}.pth.tar".format(current_version, filename))
    torch.save(state, filename)


@click.command()
def main():
    player = Player()
    dataset = SelfPlayDataset()
    improvements = 0
    current_time = int(time.time())
    for epoch in range(1, 100):
        print("--- NEW EPOCH : %d ---" % epoch)
        start_time = timeit.default_timer()
        play.create_matches(player, dataset=dataset, cores=PARRALEL_SELF_PLAY)
        total_time = timeit.default_timer() - start_time
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print("DATASET LENGTH: ", len(dataloader))
        print('Dataset ready in: ---- %.3f seconds for %d self-play games ----' % (
            total_time, SELF_PLAY_MATCH))
        print('Average time per game: ---- %.3f seconds with %d processes ----' %
                (total_time / SELF_PLAY_MATCH, PARRALEL_SELF_PLAY))
        
        if epoch > 10:
            new_player, optimizer = train.train(dataloader, deepcopy(player), 15)
        else:
            new_player, optimizer = train.train(dataloader, deepcopy(player), 5)
        if evaluate(player, new_player):
            improvements += 1
            player = new_player
            save_checkpoint(player.value_net, optimizer, improvements,\
                                     'value_net', current_time)
            save_checkpoint(player.policy_net, optimizer, improvements,\
                                     'policy_net', current_time)
            save_checkpoint(player.extractor, optimizer, improvements,\
                                     'extractor', current_time)
        print("Number of improvements so far: %d" % improvements)
        print("--- DONE WITH EPOCH ---\n\n")


if __name__ == "__main__":
    main()