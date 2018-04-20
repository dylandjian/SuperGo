from pymongo import MongoClient
from lib.dataset import SelfPlayDataset
from torch.utils.data import DataLoader
import pickle
from const import *
from torch.autograd import Variable



if __name__ == "__main__":
    client = MongoClient()
    collection = client.superGo['1524225535']
    dataset = SelfPlayDataset()
    last_id = 0
    new_games = collection.find({"id": {"$gt": last_id}}).sort('_id', -1)
    added_moves = 0
    added_games = 0
    print("[TRAIN] Fetching: %d new games from the db"% (new_games.count()))

    for game in new_games:
        number_moves = dataset.update(pickle.loads(game['game']))
        added_moves += number_moves
        added_games += 1
        break

        ## You cant replace more than 40% of the dataset at a time
        # if added_moves > MOVES * 0.4:
        #     break
    
    print("[TRAIN] Last id: %d, added games: %d, added moves: %d"\
                    % (last_id, added_games, added_moves))
    
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=1)
    for batch_idx, (state, move, winner) in enumerate(dataloader):
        x = Variable(state).type(DTYPE_FLOAT)
        assert 0
