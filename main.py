import multiprocessing
import time
import signal
import click
import os
from lib.train import train
from lib.play import play, self_play
from lib.process import MyPool


@click.command()
@click.option("--folder", default=-1)
@click.option("--ite", default=False)
def main(folder, ite):
    multiprocessing.set_start_method('spawn')
    if folder == -1:
        current_time = str(int(time.time()))
    else:
        current_time = str(folder)
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = MyPool(2)
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        x = pool.apply_async(self_play, args=(current_time, ite,))
        y = pool.apply_async(train, args=(current_time, ite,))
        x.get()
        # y.get()
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
    pool.join()

if __name__ == "__main__":
    main()


