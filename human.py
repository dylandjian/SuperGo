#!/home/dylan/.virtualenvs/superGo/bin/python

from lib.play import play, Game
import click
from lib.utils import get_player
from const import *
from lib.gtp import Engine
import os


def load_player(folder, ite):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                        'saved_models')
    if folder == -1:
        folders = os.listdir(path)
        if len(folders) > 0:
            folder = folders[-1]
        else:
            return "[GTP] No folders inside saved_models"
    elif not os.path.isdir(os.path.join(path, folder)):
        return "[GTP] The folder doesnt seem to exist !"

    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)
    if ite == -1:
        if len(files) > 0:
            all_ite = list(map(lambda x: int(x.split('-')[0]), files))
            files.sort()
            last_ite = all_ite[-1]
        else:
            return "[GTP] No files inside the folder !"
    else:
        file_ite = "{}-extractor.pth.tar".format(ite)
        if not os.path.isfile(os.path.join(folder_path, file_ite)):
            return "[GTP] The specified iteration doesn't exist"

    return get_player(folder, last_ite)


@click.command()
@click.option("--folder", default=-1)
@click.option("--ite", default=-1)
@click.option("--gtp/--no-gtp", default=False)
def main(folder, ite, gtp):
    player = load_player(folder, ite)
    if not isinstance(player, str):
        game = Game(player, 0)
        engine = Engine(game, board_size=game.goban_size)

        while True:
            print(engine.send(input()))
    elif not gtp:
        print(player)
    else:
        print("¯\_(ツ)_/¯")


if __name__ == "__main__":
    main()