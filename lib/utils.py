import os
from models.agent import Player


def get_ite(folder_path, ite):
    """ Either get the last iteration of 
        the specific folder or verify it ite exists """

    if int(ite) == -1:
        files = os.listdir(folder_path)
        if len(files) > 0:
            all_ite = list(map(lambda x: int(x.split('-')[0]), files))
            all_ite.sort()
            file_ite = all_ite[-1]
        else:
            return False
    else:
        test_file = "{}-extractor.pth.tar".format(ite)
        if not os.path.isfile(os.path.join(folder_path, test_file)):
            return False
        file_ite = ite
    return file_ite


def load_player(folder, ite):
    """ Load a player given a folder and an iteration """

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
    last_ite = get_ite(folder_path, ite)
    if not last_ite:
        return False, False

    return get_player(folder, int(last_ite))


def get_player(current_time, improvements):
    """ Load the models of a specific player """

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                            '..', 'saved_models', str(current_time))
    try:
        mod = os.listdir(path)
        models = list(filter(lambda model: (model.split('-')[0] == str(improvements)), mod))
        models.sort()
        if len(models) == 0:
            return False, improvements
    except FileNotFoundError:
        return False, improvements
    
    player = Player()
    player.load_models(path, models)
    return player, improvements + 1
