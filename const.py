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
PARRALEL_SELF_PLAY = 1
## Number of evaluation parralel games 
PARRALEL_EVAL = 2
## MCTS parallel
MCTS_PARRALEL = 1
####


##### GLOBAL

## Size of the Go board
GOBAN_SIZE = 9
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
MOMENTUM = 0.92
## Activate MCTS
NO_MCTS = True

#####


##### SELF-PLAY

## Number of self-play before training
SELF_PLAY_MATCH = 40
## Number of matches to run per process
NUM_MATCHES = SELF_PLAY_MATCH // PARRALEL_SELF_PLAY

#####


##### TRAINING

## Number of moves to consider when creating the batch
MOVES = 10000
## Number of mini-batch before evaluation during training
BATCH_SIZE = 128
## Number of channels of the output feature maps
OUTPLANES_MAP = 10
## Shape of the input state
INPLANES = (HISTORY + 1) * 2 + 1
## Probabilities for all moves + pass
OUTPLANES = (GOBAN_SIZE ** 2) + 1
## Number of residual blocks
BLOCKS = 10
## Number of training step before evaluating
TRAIN_STEPS = 200
## Optimizer
ADAM = False
## Learning rate annealing factor
LR_DECAY = 0.1
## Learning rate anmnealing interval
LR_DECAY_ITE = 1 * TRAIN_STEPS

#####


##### EVALUATION

## Number of matches against its old version to evaluate
## the newly trained network
EVAL_MATCHS = 50
## Threshold to keep the new neural net
EVAL_THRESH = 0.53


#####