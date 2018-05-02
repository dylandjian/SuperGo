from .play import play
from const import *


def evaluate(player, new_player):
    """ Used to evaluate the best network against 
        the newly trained model """ 

    print("[EVALUATION] Starting to evaluate trained model !")
    results = play(player, opponent=new_player)
    black_wins = 0
    white_wins = 0
    for result in results:
        if result[0] == 1:
            white_wins += 1
        else:
            black_wins += 1
    
    print("[EVALUATION] black wins: %d vs %d for white"\
             % (black_wins, white_wins))
    if black_wins >= EVAL_THRESH * len(results):
        return True
    return False
