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
* Better display for game
* Statistics
* Just playing with already trained models with user input
* OGS API ?


# DONE

* Multiprocessing of games
* Models and training without MCTS
* Evaluation
* Dataset ring buffer of self-play games
