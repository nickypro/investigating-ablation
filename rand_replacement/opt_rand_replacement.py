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

from rand_inputs import random_characters

c = PruningConfig(
    model_repo = "facebook/opt-1.3b",
    token_limit= 1000,
    dtype = "fp16",
    wandb_run_name = "opt-1.3b 5% resampling rand chars",
    wandb_project = "nicky-resampling-testing",
    focus = "pile",
    cripple="code",
    collection_sample_size = 1e4,
    eval_sample_size = 1e5,
    ff_frac = 0.00,
    attn_frac = 0.05,
    n_steps = 20,
)

m = Model(c.model_repo, limit=c.token_limit, dtype=c.dtype)
run_random_replacement(m, c, random_characters)
