from lib.play import play
from models.agent import player
import click
from const import *
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
            last_ite = int(files[-1].split('-')[0])
        else:
            return "[GTP] No files inside the folder !"
    else:
        file_ite = "{}-extractor.pth.tar".format(ite)
        if not os.path.isfile(os.path.join(folder_path, file_ite)):
            return "[GTP] The specified iteration doesn't exist"

    return get_player(folder, ite)


@click.command()
@click.option("--folder", default=-1)
@click.option("--ite", default=-1)
@click.option("--gtp", default=False)
def main(folder, ite, gtp)
    player = load_player(folder, ite, gtp)
    if not isinstance(player, str):

    
    elif not gtp:
        print(player)
    else:
        print("¯\_(ツ)_/¯")


if __name__ == "__main__":
    main()