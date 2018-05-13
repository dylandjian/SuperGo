import torch
import multiprocessing

##### CONFIG

## CUDA variable from Torch
CUDA = torch.cuda.is_available()
## Dtype of the tensors depending on CUDA
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")
## Number of self-play parallel games
PARALLEL_SELF_PLAY = 2
## Number of evaluation parallel games 
PARALLEL_EVAL = 3
## MCTS parallel
MCTS_PARALLEL = 8


##### GLOBAL

## Size of the Go board
GOBAN_SIZE = 9
## Number of move to end a game
MOVE_LIMIT = GOBAN_SIZE ** 2 * 2.5
## Maximum ratio that can be replaced in the rotation buffer
MAX_REPLACEMENT = 0.4
## Number of last states to keep
HISTORY = 7
## Learning rate
LR = 0.01
## Number of MCTS simulation
MCTS_SIM = 64
## Exploration constant
C_PUCT = 0.2
## L2 Regularization
L2_REG = 0.0001
## Momentum
MOMENTUM = 0.9
## Activate MCTS
MCTS_FLAG = True
## Epsilon for Dirichlet noise
EPS = 0.25
## Alpha for Dirichlet noise
ALPHA = 0.03
## Batch size for evaluation during MCTS
BATCH_SIZE_EVAL = 2
## Number of self-play before training
SELF_PLAY_MATCH = PARALLEL_SELF_PLAY
## Number of moves before changing temperature to stop
## exploration
TEMPERATURE_MOVE = 5 


##### TRAINING

## Number of moves to consider when creating the batch
MOVES = 2000
## Number of mini-batch before evaluation during training
BATCH_SIZE = 64
## Number of channels of the output feature maps
OUTPLANES_MAP = 10
## Shape of the input state
INPLANES = (HISTORY + 1) * 2 + 1
## Probabilities for all moves + pass
OUTPLANES = (GOBAN_SIZE ** 2) + 1
## Number of residual blocks
BLOCKS = 10
## Number of training step before evaluating
TRAIN_STEPS = 6 * BATCH_SIZE
## Optimizer
ADAM = False
## Learning rate annealing factor
LR_DECAY = 0.1
## Learning rate annnealing interval
LR_DECAY_ITE = 100 * TRAIN_STEPS
## Print the loss
LOSS_TICK = BATCH_SIZE // 4
## Refresh the dataset
REFRESH_TICK = BATCH_SIZE


##### EVALUATION

## Number of matches against its old version to evaluate
## the newly trained network
EVAL_MATCHS = 20
## Threshold to keep the new neural net
EVAL_THRESH = 0.55
