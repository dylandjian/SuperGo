import multiprocessing
import multiprocessing.pool
from .game import Game


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


class GameManager(multiprocessing.Process):
    """
    Used to manage a Queue of process. In charge of the interaction
    between the processes.
    """

    def __init__(self, game_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.game_queue = game_queue
        self.result_queue = result_queue


    def run(self):
        """ Execute a task from the game_queue """

        process_name = self.name
        while True:
            try:
                next_task = self.game_queue.get(600000)

                ## End the processes that are done
                if next_task is None:
                    self.game_queue.task_done()
                    break

                answer = next_task()
                self.game_queue.task_done()
                self.result_queue.put(answer)
            except Exception as e:
                print("Game has thrown an error")


def create_matches(player, opponent=None, cores=1, match_number=10):
    """ Create the process queue """

    queue = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    game_results = []

    game_managers = [
        GameManager(queue, results)
        for _ in range(cores)
    ]

    for game_manager in game_managers:
        game_manager.start()

    for id in range(match_number):
        queue.put(Game(player, id, opponent=opponent))
    
    for _ in range(cores):
        queue.put(None)
    
    return queue, results


