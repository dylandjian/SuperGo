#!/home/dylan/.virtualenvs/superGo/bin/python

import numpy as np
import re
import pickle
import sys
import click
from lib.gtp import format_success, parse_message
from pymongo import MongoClient


def game_to_gtp(game):
    """ Take a game from the database and convert it to send GTP instructions """

    moves = np.array(game[0])[:,[1,2]]
    boardsize = int(np.sqrt(moves[0][0].shape[0]))
    move_count = 0

    ## Wait for input
    while True:
        try:
            message_id, command, arguments = parse_message(input())
        except EOFError:
            break

        if "genmove" in command:
            if move_count < moves.shape[0]:
                move = np.where(moves[move_count][0] == 1.0)[0][0]
                if move == boardsize ** 2:
                    print(format_success(None, response="pass"))
                else:
                    print(format_success(None, response="{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"\
                            [int(move % boardsize)], int(boardsize - move // boardsize))))
                move_count += 1
            else:
                print('?name    %s    ???\n\n' % (command))
        elif "name" in command:
            print(format_success(None, response="test"))
        else:
            print('?name    %s    ???\n\n' % (command))


@click.command()
@click.option("--folder", default=-1)
def main(folder):
    ## Init Mongo
    client = MongoClient()
    db = client.superGo

    ## Get latest bot version
    if folder == -1:
        collection = list(db.collection_names())
        collection.sort()
        collection = collection[-1]
    else:
        collection = str(folder)

    if collection:
        game_collection = db[collection]

        ## Get the latest game
        last_game = game_collection.find().sort('_id', -1).limit(2).next()
        final_game = pickle.loads(last_game['game'])
        game_to_gtp(final_game)

if __name__ == "__main__":
    main()
