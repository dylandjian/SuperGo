from lib.play import create_matches, Game
from models.agent import Player
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    player = Player()
    opponent = Player()
    for i in range(5):
        game = Game(player, 1, mcts_flag=True)
        game()
        # for j in range(1):
        #     result = results.get()
        #     print('done: %d' % j)
        # queue.close()