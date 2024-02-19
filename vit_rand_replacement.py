from taker import Model
from taker.eval import evaluate_all
from taker.data_classes import PruningConfig, RunDataHistory
from taker.prune import prune_random
from tqdm import tqdm
import wandb

def run_random_replacement(m: Model, c: PruningConfig, sampled_text: str):
    # Prepare data logging
    history = RunDataHistory(c.datasets)
    wandb.init(
        project=c.wandb_project,
        entity=c.wandb_entity,
        name=c.wandb_run_name,
        )
    wandb.config.update(c.to_dict(), allow_val_change=True)

    # Get the random activations we will be replacing with
    preout_rand = m.get_attn_pre_out_activations(sampled_text)

    # evaluate the modified model
    if c.run_pre_test:
        data = evaluate_all(m, c.eval_sample_size,
            c.datasets, c.collection_sample_size)
        history.add(data)
        print(history.df.T)

    # run all the steps of pruning
    ff_pruned, attn_pruned = None, None
    for i in tqdm(range(c.n_steps)):
        # prune some neurons
        ff_pruned, attn_pruned, misc = \
            prune_random(m, attn_frac=c.attn_frac, ff_frac=c.ff_frac, ff_pruned=ff_pruned, attn_pruned=attn_pruned)

        # update the pruned neurons to be resampled
        preout_rand_masked = m.run_inverse_masking(preout_rand, "attn_pre_out")
        m.update_actadd(preout_rand_masked.reshape(m.cfg.n_layers, m.limit, m.cfg.d_model), "attn_pre_out")

        # evaluate the modified model
        data = evaluate_all(m, c.eval_sample_size,
            [c.focus], c.collection_sample_size)
        history.add(data)

    wandb.finish()
    return history

import numpy as np

rand_image = np.random.randint(0, 255, [3, 244, 244])
print(m.processor(rand_image)["pixel_values"][0].shape)

c = PruningConfig(
    model_repo = "google/vit-base-patch16-224",
    token_limit= 1000,
    dtype = "fp32",
    wandb_run_name = "vit 10% resampling rand pixels",
    wandb_project = "nicky-resampling-testing",
    focus = "imagenet-1k",
    cripple="imagenet-1k-birds",
    collection_sample_size = 1e4,
    eval_sample_size = 1e4,
    ff_frac = 0.00,
    attn_frac = 0.10,
    n_steps = 10,
)

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

m = Model(c.model_repo, limit=c.token_limit, dtype=c.dtype)
run_random_replacement(m, c, rand_image)
