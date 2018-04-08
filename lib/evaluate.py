from .play import play
from const import *


def evaluate(player, new_player):
    results = play(player, opponent=new_player)
    black_wins = 0
    white_wins = 0
    for result in results:
        if result[0] == 0:
            white_wins += 1
        else:
            black_wins += 1
    
    print("[EVAL] black wins: %d vs %d for white" % (black_wins, white_wins))
    if black_wins >= EVAL_THRESH * len(results):
        return True
    return False