import pickle
import time
from copy import deepcopy
from const import *
from pymongo import MongoClient
from .utils import get_player, load_player, _prepare_state
from .process import GameManager, create_matches



def self_play(current_time, loaded_version):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    client = MongoClient()
    collection = client.superGo[current_time]
    game_id = 0
    current_version = 1
    player = False

    while True:

        ## Load the player when restarting traning
        if loaded_version:
            new_player, checkpoint = load_player(current_time, 
                                                loaded_version)
            game_id = collection.find().count()
            current_version = checkpoint['version'] + 1
            loaded_version = False
        else:
            new_player, checkpoint = get_player(current_time, current_version)
            if new_player:
                current_version = checkpoint['version'] + 1

        print("[PLAY] Current improvement level: %d" % current_version)
        if current_version == 1 and not player and not new_player:
            print("[PLAY] Waiting for first player")
            time.sleep(5)
            continue

        if new_player:
            player = new_player
            print("[PLAY] New player !")

        queue, results = create_matches(player , \
                    cores=PARALLEL_SELF_PLAY, match_number=SELF_PLAY_MATCH) 
        print("[PLAY] Starting to fetch fresh games")
        try:
            queue.join()
            for _ in range(SELF_PLAY_MATCH):
                result = results.get()
                if result:
                    collection.insert({
                        "game": result,
                        "id": game_id
                    })
                    game_id += 1
            print("[PLAY] Done fetching")
        finally:
            queue.close()
            results.close()


def play(player, opponent):
    """ Game between two players, for evaluation """

    queue, results = create_matches(deepcopy(player), opponent=deepcopy(opponent), \
                cores=PARALLEL_EVAL, match_number=EVAL_MATCHS) 
    try:
        queue.join()
        
        print("[EVALUATION] Starting to fetch fresh games")
        final_result = []
        for idx in range(EVAL_MATCHS):
            result = results.get()
            if result:
                final_result.append(pickle.loads(result))
        print("[EVALUATION] Done fetching")
    finally:
        queue.close()
        results.close()
    return final_result
