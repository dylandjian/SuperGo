from lib.train import train
from lib.play import play, self_play
from lib.process import MyPool
import multiprocessing
import time
import signal
import click




@click.command()
def main():
    multiprocessing.set_start_method('spawn')
    current_time = str(int(time.time()))
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = MyPool(2)
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        x = pool.apply_async(self_play, args=(current_time,))
        y = pool.apply_async(train, args=(current_time,))
        # x.get()
        y.get()
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
    pool.join()
    print("done")

if __name__ == "__main__":
    main()


