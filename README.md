# SuperGo

A student implementation of AlphaGo Zero paper with documentation.

Ongoing project.

# TODO (in order of priority)

* MCTS
  * Rotation of board for more training samples
  * Dirichlet noise to prior probabilities in the rootnode
  * Multiprocessing of search
* Learning rate annealing (see [this](https://discuss.pytorch.org/t/adaptive-learning-rate/320/26))
* Use logging instead of prints
* Adaptative temperature (close to 1 during the first 30 moves of self-play, close to 0 after and during evaluation)
* Optimization ?
* Better Komi ?
* File of constants that match the paper constants
* OGS / KGS API

# CURRENTLY DOING

* GTP on trained models
* Testing on 9x9 without MCTS to see if it actually learns something

# LONG TERM PLAN ?

* Statistics
* Tromp Taylor scoring ?
* Resignation ?
* Training on a big computer / server once everything is ready ?

# DONE

* Better display for game (viewer.py, converting games into GTP and then using Sabaki)
* Make the 3 components (self-play, training, evaluation) asynchronous
* Multiprocessing of games for self-play and evaluation
* Models and training without MCTS
* Evaluation
* Dataset ring buffer of self-play games
* Loading saved models
* Database for self-play games
