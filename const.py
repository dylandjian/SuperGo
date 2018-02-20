import torch
import multiprocessing

##### CONFIG

## CUDA variable from Torch
CUDA = torch.cuda.is_available()
## Number of threads, used for parallel matching atm
CPU_CORES = multiprocessing.cpu_count() * 3
#####


##### GLOBAL

## Size of the Go board
GOBAN_SIZE = 7
## Number of last states to keep
HISTORY = 7
## Learning rate
LR = 1e-2
## Number of epochs
EPOCHS = 100
## Number of MCTS simulation
MCTS_LOOK = 20

#####


##### SELF-PLAY

## Number of self-play before training
SELF_PLAY_MATCH = 200 
NUM_MATCHES = SELF_PLAY_MATCH// CPU_CORES

#####


##### TRAINING

## Size of mini batch of moves during learning
MINIBATCH = 64
## Number of moves to consider when creating the batch
MOVES = 5000
## Number of mini-batch before evaluation during training
TRAIN_EXAMPLE = 32 
## Number of channels of the output feature maps
OUTPLANES_MAP = 10

#####


##### EVALUATION

## Number of matches against its old version to evaluate
## the newly trained network
EVAL_MATCHS = 10
## Threshold to keep the new neural net
EVAL_THRESH = 0.6


#####