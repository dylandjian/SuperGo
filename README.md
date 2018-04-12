# SuperGo

A student implementation of AlphaGo Zero paper with documentation.

Ongoing project.

# TODO (in order of priority)

* MCTS
  * Tree search
  * Rotation of board for more training samples
  * Adaptative temperature (close to 1 during the first 30 moves of self-play, close to 0 after and during evaluation)
  * Dirichlet noise to prior probabilities in the rootnode
  * Multiprocessing of search
* Better Komi ?
* File of constants that match the paper constants
* OGS / KGS API
* Use logging instead of prints

# CURRENTLY DOING

* Resume training
* Testing on 9x9 without MCTS to see if it actually learns something

# LONG TERM PLAN ?

* Statistics
* Optimization ?
* Tromp Taylor scoring ?
* Resignation ?
* Training on a big computer / server once everything is ready ?

# DONE

* GTP on trained models
* Learning rate annealing (see [this](https://discuss.pytorch.org/t/adaptive-learning-rate/320/26))
* Better display for game (viewer.py, converting self-play games into GTP and then using Sabaki)
* Make the 3 components (self-play, training, evaluation) asynchronous
* Multiprocessing of games for self-play and evaluation
* Models and training without MCTS
* Evaluation
* Dataset ring buffer of self-play games
* Loading saved models
* Database for self-play games
