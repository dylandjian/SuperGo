#!/home/dylan/.virtualenvs/superGo/bin/python

import numpy as np
import re
import pickle
import sys
import click
from lib.gtp import format_success, parse_message
from pymongo import MongoClient


def game_to_gtp(game, game_id, collection_name, color):
    """ Take a game from the database and convert it to send GTP instructions """

    board_size = int(np.sqrt(len(game[0][0][1]) - 1))
    moves = np.array(game[0])[:,3]
    move_count = 0 if color == 0 else 1
    game_winner = game[1]

    ## Wait for input
    while True:
        try:
            message_id, command, arguments = parse_message(input())
        except EOFError:
            break

        if "genmove" in command:
            if move_count < moves.shape[0]:
                move = moves[move_count]
                if move == board_size ** 2:
                    print(format_success(None, response="pass"))
                else:
                    print(format_success(None, response="{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"\
                            [int(move % board_size)], int(board_size - move // board_size))))
                move_count += 2
            else:
                print('?name    %s    ???\n\n' % (command))
        elif "name" in command:
            print(format_success(None, response="folder {}, game id: {}, winner: {}"\
                                    .format(collection_name, game_id, game_winner)))
        elif "play" in command:
            print(format_success(message_id, ""))
        else:
            print('?name    %s    ???\n\n' % (command))


@click.command()
@click.option("--folder", default=-1)
@click.option("--game_id", default=-1)
@click.option("--color", default=1)
def main(folder, game_id, color):
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
        if game_id == -1:
            last_game = game_collection.find().sort('_id', -1).limit(1)
        else:
            last_game = game_collection.find({"id": game_id})
        if last_game.count() == 0:
            print("Wrong game_id or the database superGo doesnt have any collections")
        else:
            for game in last_game:
                final_game = pickle.loads(game['game'])
                game_to_gtp(final_game, game['id'], collection, color)
                break

if __name__ == "__main__":
    main()
