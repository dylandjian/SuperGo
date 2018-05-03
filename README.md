# SuperGo

A student implementation of AlphaGo Zero paper with documentation.

Ongoing project.

# TODO (in order of priority)

* File of constants that match the paper constants ?
* OGS / KGS API ?
* Use logging instead of prints ?

# CURRENTLY DOING

* Optimizations
* MCTS
  * Improving the multithreading of the search by making it work when the BATCH_SIZE_EVAL is > than the number of threads used during the MCTS
* Do something about the process leaking
* Clean code, create install script, write documentation
* Currently training on 9x9 64 simulations 10 layers ResNet on my computer, didn't seem to have learnt a lot after 4000~ games, 450k training epochs and 75 improvements ( = trained network replacing the current best network) due to the duration of the evaluation process : 2 games at a time (only have 8 cores on my computer, 1 for self-play, 1 for training, 3 for self-play games, 1 for evaluation, 2 for evaluations games) which takes approximately 30s / game so 6 to 7 minutes per evaluation.

# DONE

* Game that are longer than the threshold of moves are now used
* MCTS
  * Tree search
  * Dirichlet noise to prior probabilities in the rootnode
  * Adaptative temperature (either take max or proportionally)
  * Sample random rotation or reflection in the dihedral group
  * Multithreading of search (kinda badly implemented for now)
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
* Statistics
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

# Statistics

## For a 10 layers deep Resnet evaluated on 50 games 64 simulations

### 9x9 board

* 0.2377991s / move - 0.00371561093s / simulation 2 threads 2 batch_size_eval
* 0.1624937s / move - 0.00253896406s / simulation 4 threads 4 batch_size_eval
* 0.1465123s / move - 0.00228925468s / simulation 8 threads 8 batch_size_eval
* 0.1401098s / move - 0.00218921563s / simulation 16 threads 16 batch_size_eval

### 19x19 board

* 0.6306054s / move - 0.012612108s / simulation with 2 threads and 2 batch_size_eval with 50 simulations

# Differences with the official paper

* No resignation
* PyTorch instead of Tensorflow
* Python instead of (probably) C++ / C
