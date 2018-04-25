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
@click.option("--version", default=False)
def main(folder, version):

    ## Start method for PyTorch
    multiprocessing.set_start_method('spawn')

    ## Create folder name if not provided
    if folder == -1:
        current_time = str(int(time.time()))
    else:
        current_time = str(folder)

    ## Catch SIGNINT
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = MyPool(2)
    signal.signal(signal.SIGINT, original_sigint_handler)

    try:
        self_play_proc = pool.apply_async(self_play, args=(current_time, version,))
        train_proc = pool.apply_async(train, args=(current_time, version,))

        ## Comment one line or the other to get the stack trace
        ## Must add a loooooong timer otherwise signals are not caught
        self_play_proc.get(60000000)
        train_proc.get(60000000)

    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()


