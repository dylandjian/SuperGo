import torch
import multiprocessing

##### CONFIG

## CUDA variable from Torch
CUDA = torch.cuda.is_available()
## Dtype of the tensors depending on CUDA
DTYPE_FLOAT = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
DTYPE_LONG = torch.cuda.LongTensor if CUDA else torch.LongTensor
## Number of process, used for parallel matching atm
## Number of self-play parallel games
# PARRALEL_SELF_PLAY = multiprocessing.cpu_count() - 2
PARRALEL_SELF_PLAY = 6
## Number of evaluation parralel games 
PARRALEL_EVAL = 5
## MCTS parallel
MCTS_PARRALEL = 2
####


##### GLOBAL

## Size of the Go board
GOBAN_SIZE = 13
## Number of last states to keep
HISTORY = 7
## Learning rate
LR = 0.01
## Number of epochs
EPOCHS = 100
## Number of MCTS simulation
MCTS_LOOK = 200
## Temperature
TEMP = 2
## Exploration constant
C_PUCT = 0.2
## L2 Regularization
L2_REG = 0.0001
## Momentum
MOMENTUM = 0.9
## Activate MCTS
NO_MCTS = True

#####


##### SELF-PLAY

## Number of self-play before training
SELF_PLAY_MATCH = 50
## Number of matches to run per process
NUM_MATCHES = SELF_PLAY_MATCH // PARRALEL_SELF_PLAY

#####


##### TRAINING

## Number of moves to consider when creating the batch
MOVES = 30000
## Number of mini-batch before evaluation during training
BATCH_SIZE = 512
## Number of channels of the output feature maps
OUTPLANES_MAP = 10
## Shape of the input state
INPLANES = (HISTORY + 1) * 2 + 1
## Probabilities for all moves + pass
OUTPLANES = (GOBAN_SIZE ** 2) + 1
## Number of residual blocks
BLOCKS = 7

#####


##### EVALUATION

## Number of matches against its old version to evaluate
## the newly trained network
EVAL_MATCHS = 100
## Threshold to keep the new neural net
EVAL_THRESH = 0.53


#####