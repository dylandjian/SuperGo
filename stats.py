from lib.game import Game
from lib.process import create_matches
from models.agent import Player
import multiprocessing
import timeit
import pickle
import numpy as np
from const import PARALLEL_SELF_PLAY, MCTS_PARALLEL, MCTS_SIM, BATCH_SIZE_EVAL 
from subprocess import call


MCTS_PARALLELS = [2, 4, 6, 8, 16]
MCTS_SIMS = [32, 64, 128, 160]
BATCH_SIZE_EVALS = [2, 4, 6, 8]
SAMPLE_NUM = 30


def overwrite_file(old_values, new_values):
    for idx, new_value in new_values.items():
        regex = "s/{}\ =\ {}/{}\ =\ {}/g".format(idx, old_values[idx],
                idx, new_value)
        sub = call(['sed', '-i', regex, 'const.py'])


def do_sims(player, old_values, mcts_parallel=2, mcts_sim=8, batch_size_eval=2):
    new_values = {
        "MCTS_PARALLEL": mcts_parallel,
        "MCTS_SIM": mcts_sim,
        "BATCH_SIZE_EVAL": batch_size_eval
    }
    overwrite_file(old_values, new_values)
    queue, results = create_matches(player, cores=PARALLEL_SELF_PLAY, 
                    opponent=player, match_number=SAMPLE_NUM)
    moves = []
    times = []
    try:
        queue.join()
        for _ in range(SAMPLE_NUM):
            result = pickle.loads(results.get())
            moves.append(result[1])
            times.append(result[2])
    finally:
        queue.close()
        results.close()
    
    print("-- FINAL RESULTS FOR %d GAMES WITH MCTS PARALLEL %d SIMS %d BATCH_SIZE %d --"\
            % (SAMPLE_NUM, mcts_parallel, mcts_sim, batch_size_eval))
    print("total game duration: %d seconds, total game move count: %d" \
                % (sum(times) / PARALLEL_SELF_PLAY, sum(moves)))
    print("average game duration: %.5f seconds, average game move count: %.1f" \
                % (np.mean(times), np.mean(moves)))
    print("average move duration: %.5f seconds, average sim duration: %.8f seconds" \
            % (sum(times) / sum(moves),  sum(times) / (sum(moves) * mcts_sim)))
    print("-- DONE --\n")
    return new_values


def stats_report():
    multiprocessing.set_start_method("spawn")
    player = Player()
    old_values = {
        "MCTS_PARALLEL": MCTS_PARALLEL,
        "MCTS_SIM": MCTS_SIM,
        "BATCH_SIZE_EVAL": BATCH_SIZE_EVAL
    }

    old_values = do_sims(player, old_values, mcts_parallel=2, mcts_sim=32, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=4, mcts_sim=32, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=6, mcts_sim=32, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=8, mcts_sim=32, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=12, mcts_sim=32, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=8, mcts_sim=32, batch_size_eval=4)
    old_values = do_sims(player, old_values, mcts_parallel=12, mcts_sim=32, batch_size_eval=4)
    old_values = do_sims(player, old_values, mcts_parallel=12, mcts_sim=32, batch_size_eval=6)
    
    old_values = do_sims(player, old_values, mcts_parallel=2, mcts_sim=128, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=4, mcts_sim=128, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=6, mcts_sim=128, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=8, mcts_sim=128, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=12, mcts_sim=128, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=8, mcts_sim=128, batch_size_eval=4)
    old_values = do_sims(player, old_values, mcts_parallel=12, mcts_sim=128, batch_size_eval=4)
    old_values = do_sims(player, old_values, mcts_parallel=12, mcts_sim=128, batch_size_eval=6)
    
    old_values = do_sims(player, old_values, mcts_parallel=2, mcts_sim=160, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=4, mcts_sim=160, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=6, mcts_sim=160, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=8, mcts_sim=160, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=12, mcts_sim=160, batch_size_eval=2)
    old_values = do_sims(player, old_values, mcts_parallel=8, mcts_sim=160, batch_size_eval=4)
    old_values = do_sims(player, old_values, mcts_parallel=12, mcts_sim=160, batch_size_eval=4)
    old_values = do_sims(player, old_values, mcts_parallel=12, mcts_sim=160, batch_size_eval=6)

    


if __name__ == "__main__":
    stats_report()