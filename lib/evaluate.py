from .play import create_matches
from const import *


def evaluate(player, new_player):
    results = create_matches(player, opponent=new_player, \
                    match_number=EVAL_MATCHS, cores=PARRALEL_EVAL)
    black_wins = 0
    white_wins = 0
    for result in results:
        if result[0] == 0:
            white_wins += 1
        else:
            black_wins += 1
    
    print("black wins: %d vs %d for white" % (black_wins, white_wins))
    if black_wins >= EVAL_THRESH * len(results):
        print("USING NEW AGENT !!!")
        return True
    print("USING OLD AGENT STILL !!!")
    return False