#from separability.data_classes import PruningConfig
#from separability.parser import cli_parser
#from separability.prune import run_pruning
import sys

sys.path.append('/home/ubuntu/taker-ben/src/taker')
import torch

#from taker.data_classes import PruningConfig
#from taker.parser import cli_parser
#from taker.prune import run_pruning
#import torch
from taker.data_classes import PruningConfig
from taker.parser import cli_parser
from prune import run_pruning

# Configure initial model and tests
c = PruningConfig(
    wandb_project = "testing",
    model_repo   = "facebook/opt-1.3b",
    token_limit  = 1000,
    run_pre_test = True,
    # Removals parameters
    ff_frac   = 0.02,
    ff_eps    = 0.001,
    ff_offset_mode = "zero",
    attn_offset_mode = "zero",
    attn_frac = 0.00,
    attn_eps  = 1e-4,
    focus     = "pile_codeless",
    cripple   = "code",
    additional_datasets=tuple(),
    recalculate_activations = True, # iterative vs non-iterative pruning
)

# Parse CLI for arguments
c, args = cli_parser(c)


import torch
import numpy as np
import random
def set_seed(seed):
    random.seed(seed)               # Python random module.
    np.random.seed(seed)            # Numpy module.
    torch.manual_seed(seed)         # PyTorch random number generator for CPU.
    torch.cuda.manual_seed(seed)    # PyTorch random number generator for CUDA.
    torch.backends.cudnn.deterministic = True  # To ensure that CUDA selects deterministic algorithms.
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)


# Run the iterated pruning
with torch.no_grad():
    model, history = run_pruning(c)
