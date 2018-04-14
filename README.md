# SuperGo

A student implementation of AlphaGo Zero paper with documentation.

Ongoing project.

# TODO (in order of priority)

* Better Komi ?
* File of constants that match the paper constants
* OGS / KGS API
* Use logging instead of prints ?

# CURRENTLY DOING

* MCTS
  * Tree search
  * Rotation of board for more training samples
  * Adaptative temperature (close to 1 during the first 30 moves of self-play, close to 0 after and during evaluation)
  * Dirichlet noise to prior probabilities in the rootnode
  * Multiprocessing of search

# LONG TERM PLAN ?

* Compile my own version of Sabaki to watch games automatically
* Statistics
* Optimization ?
* Tromp Taylor scoring ?
* Resignation ?
* Training on a big computer / server once everything is ready ?

# DONE

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
