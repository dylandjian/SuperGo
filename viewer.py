#!/home/dylan/.virtualenvs/superGo/bin/python

from pymongo import MongoClient
import numpy as np
import re
import pickle
import sys


class GTP:
    def __init__(self, name, version, komi, boardsize):
        self.name = name
        self.version = version
        self.komi = komi
        self.boardsize = boardsize

    def send_cmd(self, cmd):
        print("{} {}".format(cmd, str(getattr(self, cmd))), end="")
        sys.stdout.flush()
    
    def send(self, msg):
        print("{}".format(msg), end="")
        sys.stdout.flush()


def pre_engine(s):
    s = re.sub("[^\t\n -~]", "", s)
    s = s.split("#")[0]
    s = s.replace("\t", " ")
    return s


def parse_message(message):
    message = pre_engine(message).strip()
    first, rest = (message.split(" ", 1) + [None])[:2]
    if first.isdigit():
        message_id = int(first)
        if rest is not None:
            command, arguments = (rest.split(" ", 1) + [None])[:2]
        else:
            command, arguments = None, None
    else:
        message_id = None
        command, arguments = first, rest

    return message_id, command, arguments


def format_success(message_id, response=None):
    if response is None:
        response = ""
    else:
        response = " {}".format(response)
    if message_id:
        return "={}{}\n\n".format(message_id, response)
    else:
        return "={}\n\n".format(response)

def game_to_gtp(game):
    moves = np.array(game[0])[:,[1,2]]
    boardsize = np.sqrt(np.max(moves[:,0]))
    current_move = 0
    while True:
        try:
            message_id, command, arguments = parse_message(input())
        except EOFError:
            break
        if "genmove" in command:
            if current_move < moves.shape[0]:
                x = moves[current_move][0]
                if x == boardsize ** 2:
                    print(format_success(None, response="pass"))
                else:
                    print(format_success(None, response="{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[int(x % boardsize)], int(boardsize - x // boardsize))))
                current_move += 1
        elif "name" in command:
            print(format_success(None, response="test"))
        else:
            print('?name    %s    ???\n\n' % (command))
        # sys.stdout.flush()


if __name__ == "__main__":
    ## Init Mongo
    client = MongoClient()
    db = client.superGo

    ## Get latest bot version
    collection = list(db.collection_names())
    collection.sort()
    collection = collection[-1]
    if collection:
        game_collection = db[collection]

        # while True:
        ## Get the latest game
        last_game = game_collection.find().sort('_id', -1).limit(2).next()
        final_game = pickle.loads(last_game['game'])
        game_to_gtp(final_game)
        # break
