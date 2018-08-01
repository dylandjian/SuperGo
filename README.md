# SuperGo

A student implementation of AlphaGo Zero paper with documentation.

Ongoing project.

# TODO (in order of priority)

* Do something about the process leaking
* File of constants that match the paper constants ?
* OGS / KGS API ?
* Use logging instead of prints ?

# CURRENTLY DOING

* Optimizations
* Clean code, create install script, write documentation
* Trying to see if it learns something on my computer

# DONE

* Statistics (branch statistics)
* Game that are longer than the threshold of moves are now used
* MCTS
  * Tree search
  * Dirichlet noise to prior probabilities in the rootnode
  * Adaptative temperature (either take max or proportionally)
  * Sample random rotation or reflection in the dihedral group
  * Multithreading of search
  * Batch size evaluation to save computation
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
* Tromp Taylor scoring
* Dataset ring buffer of self-play games
* Loading saved models
* Database for self-play games

# LONG TERM PLAN ?

* Compile my own version of Sabaki to watch games automatically while traning
* Resignation ?
* Training on a big computer / server once everything is ready ?

# Resources

* [The article for this code](https://github.com/dylandjian/SuperGo)
* [Official AlphaGo Zero paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
* Custom environment implementation using [pachi_py](https://github.com/openai/pachi-py/tree/master/pachi_py) following the implementation that was originally made on [OpenAI Gym](https://github.com/openai/gym/blob/6af4a5b9b2755606c4e0becfe1fc876d33130526/gym/envs/board_game/go.py)
* Using [PyTorch](https://github.com/pytorch/pytorch) for the neural networks
* Using [Sabaki](https://github.com/SabakiHQ/Sabaki) for the GUI
* [General scheme, cool design](https://applied-data.science/static/main/res/alpha_go_zero_cheat_sheet.png)
* [Monte Carlo tree search explaination](https://int8.io/monte-carlo-tree-search-beginners-guide/)
* [Nice tree search implementation](https://github.com/blanyal/alpha-zero/blob/master/mcts.py)

# Statistics, check branch stats

## For a 10 layers deep Resnet

### 9x9 board

soon

### 19x19 board

# Differences with the official paper

* No resignation
* PyTorch instead of Tensorflow
* Python instead of (probably) C++ / C
