# SuperGo

A student implementation of AlphaGo Zero paper with documentation.

Ongoing project.

# TODO (in order of priority)

* MCTS
  * Optimization ? (already done some Numba, cant go down much further I think)
  * Multithreading of search (cant multiprocess because of virtual loss, but useless in Python) ?
* File of constants that match the paper constants
* OGS / KGS API
* Better Komi ?
* Use logging instead of prints ?

# CURRENTLY DOING

* Brainlag on loss : cross entropy or KLDiv (crossentropy - entropy) ??
* Loss doesn't decrease :( still trying to see if it learns something on 9x9 with 50 simulations !

# DONE

* MCTS
  * Tree search
  * Dirichlet noise to prior probabilities in the rootnode
  * Adaptative temperature (either take max or proportionally)
  * Sample random rotation or reflection in the dihedral group
* Dihedral group of board for more training samples
* Learning without MCTS doesnt seem to work
* Resume training
* GTP on trained models (human.py, to plug with Sabaki)
* Learning rate annealing (see [this](https://discuss.pytorch.org/t/adaptive-learning-rate/320/26))
* Better display for game (viewer.py, converting self-play games into GTP and then using Sabaki)
* Make the 3 components (self-play, training, evaluation) asynchronous
* Multiprocessing of games for self-play and evaluation
* Models and training without MCTS
* Evaluation
* Dataset ring buffer of self-play games
* Loading saved models
* Database for self-play games

# LONG TERM PLAN ?

* Compile my own version of Sabaki to watch games automatically while traning
* Statistics
* Tromp Taylor scoring ?
* Resignation ?
* Training on a big computer / server once everything is ready ?

# Resources

* [Official AlphaGo Zero paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
* Custom environment implementation using [pachi_py](https://github.com/openai/pachi-py/tree/master/pachi_py) following the implementation that was originally made on [OpenAI Gym](https://github.com/openai/gym/blob/6af4a5b9b2755606c4e0becfe1fc876d33130526/gym/envs/board_game/go.py)
* Using [PyTorch](https://github.com/pytorch/pytorch) for the neural networks
* Using [Sabaki](https://github.com/SabakiHQ/Sabaki) for the GUI
* [General scheme, cool design](https://applied-data.science/static/main/res/alpha_go_zero_cheat_sheet.png)
* [Monte Carlo tree search explaination](https://int8.io/monte-carlo-tree-search-beginners-guide/)
* [Nice tree search implementation](https://github.com/blanyal/alpha-zero/blob/master/mcts.py)
