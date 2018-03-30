from .play import create_matches
from const import *


def evaluate(player, new_player):
    results = create_matches(player, opponent=new_player, \
                    match_number=EVAL_MATCHS)
    for result in results:
        print(result)
    return True