# SuperGo
A student implementation of AlphaGo Zero paper with documentation.

Ongoing project.

# TODO

* MCTS
* Multiprocessing on the 3 components (self-play, training, evaluation)
* Learning rate annealing (see [this](https://discuss.pytorch.org/t/adaptive-learning-rate/320/26))
* Rotation of board for more training samples
* Loading saved models
* File of constants that match the paper constants
* Just playing with already trained models with user input
* Better Komi ?
* Adaptative temperature (close to 1 during the first 30 moves of self-play, close to 0 after and during evaluation)
* Dirichlet noise to prior probabilities in the rootnode


# LONG TERM PLAN ?
* Better display for game
* OGS API
* Statistics
* Tromp Taylor scoring ?
* Resignation ?
* Training on a big computer / server once everything is ready ?


# DONE

* Multiprocessing of games
* Models and training without MCTS
* Evaluation
* Dataset ring buffer of self-play games
