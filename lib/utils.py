import os
from models.agent import Player


def get_player(current_time, improvements):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                            '..', 'saved_models', current_time)
    try:
        mod = os.listdir(path)
        models = list(filter(lambda model: model.startswith(str(improvements)), \
                mod))
        models.sort()
        if len(models) == 0:
            return False
    except FileNotFoundError:
        return False
    
    player = Player()
    player.load_models(path, models)
    return player
