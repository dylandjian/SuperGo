import timeit
from .play import play
from const import *


def evaluate(player, new_player):
    """ Used to evaluate the best network against 
        the newly trained model """ 

    print("[EVALUATION] Starting to evaluate trained model !")
    start_time = timeit.default_timer()

    ## Play the matches and get the results
    results = play(player, opponent=new_player)
    final_time = timeit.default_timer() - start_time
    print("[EVALUATION] Total duration: %.3f seconds, average duration:"
            " %.3f seconds" % ((final_time, final_time / EVAL_MATCHS)))

    ## Count the number of wins for each players
    black_wins = 0
    white_wins = 0
    for result in results:
        if result[0] == 1:
            white_wins += 1
        elif result[0] == 0:
            black_wins += 1
        else:
            print("[EVALUATION] Error during evaluation")
    
    print("[EVALUATION] black wins: %d vs %d for white"\
             % (black_wins, white_wins))
    
    ## Check if the trained player (black) is better than
    ## the current best player depending on the threshold
    if black_wins >= EVAL_THRESH * len(results):
        return True
    return False
