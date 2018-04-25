import os
import numpy as np
import random
import torch
from models.agent import Player
from const import *
from torch.autograd import Variable



def _prepare_state(state):
    """
    Transform the numpy state into a PyTorch tensor with cuda
    if available
    """

    x = torch.from_numpy(np.array([state]))
    x = Variable(x).type(DTYPE_FLOAT)
    return x


def get_version(folder_path, version):
    """ Either get the last versionration of 
        the specific folder or verify it version exists """

    if int(version) == -1:
        files = os.listdir(folder_path)
        if len(files) > 0:
            all_version = list(map(lambda x: int(x.split('-')[0]), files))
            all_version.sort()
            file_version = all_version[-1]
        else:
            return False
    else:
        test_file = "{}-extractor.pth.tar".format(version)
        if not os.path.isfile(os.path.join(folder_path, test_file)):
            return False
        file_version = version
    return file_version


def load_player(folder, version):
    """ Load a player given a folder and a version """

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                   '..', 'saved_models')
    if folder == -1:
        folders = os.listdir(path)
        folders.sort()
        if len(folders) > 0:
            folder = folders[-1]
        else:
            return False, False
    elif not os.path.isdir(os.path.join(path, str(folder))):
        return False, False

    folder_path = os.path.join(path, str(folder))
    last_version = get_version(folder_path, version)
    if not last_version:
        return False, False

    return get_player(folder, int(last_version))


def get_player(current_time, version):
    """ Load the models of a specific player """

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                            '..', 'saved_models', str(current_time))
    try:
        mod = os.listdir(path)
        models = list(filter(lambda model: (model.split('-')[0] == str(version)), mod))
        models.sort()
        if len(models) == 0:
            return False, version
    except FileNotFoundError:
        return False, version
    
    player = Player()
    checkpoint = player.load_models(path, models)
    return player, checkpoint


def sample_rotation(state, num=8):
    """ Apply a certain number of random transformation to the input state """

    dh_group = [(None, None), ((np.rot90, 1), None), ((np.rot90, 2), None),
                ((np.rot90, 3), None), (np.fliplr, None), (np.flipud, None),
                (np.flipud,  (np.rot90, 1)), (np.fliplr, (np.rot90, 1))]

    random.shuffle(dh_group)

    states = []
    boards = (HISTORY + 1) * 2
    for idx in range(num):
        new_state = np.zeros((boards + 1, GOBAN_SIZE, GOBAN_SIZE,))
        new_state[:boards] = state[:boards]
        for grp in dh_group[idx]:
            for i in range(boards):
                if isinstance(grp, tuple):
                    new_state[i] = grp[0](new_state[i], k=grp[1])
                elif grp is not None:
                    new_state[i] = grp(new_state[i])
        new_state[boards] = state[boards]
        states.append(new_state)
    
    if len(states) == 1:
        return np.array(states[0])
    return np.array(states)


def formate_state(state, probas, winner):
    """ Repeat the probas and the winner to make every example identical after
        the dihedral rotation have been applied """

    probas = np.reshape(probas, (1, probas.shape[0]))
    probas = np.repeat(probas, 8, axis=0)
    winner = np.full((8, 1), winner)
    return state, probas, winner
