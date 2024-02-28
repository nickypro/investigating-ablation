prune_gal() {
    poetry run python prune_30.py facebook/galactica-1.3b \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}
prune_opt() {
    poetry run python prune_30.py facebook/opt-1.3b \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}
#TODO: delete me
prune_opt_test() {
    poetry run python prune_30.py nickypro/tinyllama-15m \
        --wandb_project bens-tests --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}
prune_vit_test() {
    poetry run python prune_30.py google/vit-base-patch16-224 \
        --wandb_project bens-tests \
        --focus imagenet-1k \
        --cripple imagenet-1k-birds \
        --dtype fp32 \
        --svd_attn False \
        --attn_mode pre-out \
        --name "vit-base-patch16-224 random attn 5%  zero-ablation -- 1k token limit, 1e3 eval 1e5 collection run 3" \
        --attn_frac 0.05 \
        --ff_frac 0.00 \
        --ff_scoring random \
        --attn_scoring random \
        --eval_sample_size 1000 \
        --collection_sample_size 100000 \
        --recalculate_activations False
}

prune_quick_test() {
    #poetry run python prune_30.py nickypro/tinyllama-15m \
    poetry run python prune_30.py facebook/opt-1.3b \
        --wandb_project bens-tests \
        --focus pile_codeless \
        --cripple code \
        --run_pre_test True \
        --svd_attn False \
        --attn_mode pre-out \
        --token_limit 1000 \
        --name "opt1.3b random attn 10% kde offset" \
        --attn_frac 0.10 \
        --ff_frac 0.00 \
        --ff_scoring random \
        --attn_scoring random \
        --eval_sample_size 10000 \
        --collection_sample_size 10000 \
        --recalculate_activations True
}
prune_pyt() {
    poetry run python prune_30.py EleutherAI/pythia-1.4b \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}

prune_roberta() {
    poetry run python prune_30.py roberta-large \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \ 
	--attn_mode pre-out --token_limit 512 --name "$@"
}

prune_vit() {
    #poetry run python /root/separability/examples/prune_30.py google/vit-base-patch16-224 \
    poetry run python prune_30.py google/vit-base-patch16-224 \
	--focus imagenet-1k-birdless --cripple imagenet-1k-birds \
        --dtype fp32 \
	--eval_sample_size 1000 --collection_sample_size 100000 \
        --wandb_project birds \
	--recalculate_activations false \
	--name "$@"
}

ben_prune_vit() {
    #poetry run python /root/separability/examples/prune_30.py google/vit-base-patch16-224 \
    poetry run python prune_30.py google/vit-base-patch16-224 \
	--focus imagenet-1k --cripple imagenet-1k-birds \
        --dtype fp32 \
	--eval_sample_size 1000 --collection_sample_size 10000 \
    --name "vit rand attn 5% mean-ablation -- 1k eval, 1e5 collection run (test)" \
    --attn_frac 0.05 \
    --ff_frac 0.00 \
    --ff_scoring random \
    --attn_scoring random \
    --attn_offset_mode mean \
    --wandb_project bens-tests \
	--recalculate_activations False \
	#--name "$@"
}

nicky_prune_vit() {
    #poetry run python /root/separability/examples/prune_30.py google/vit-base-patch16-224 \
    poetry run python prune_30.py google/vit-base-patch16-224 \
	--focus imagenet-1k --cripple imagenet-1k-birds \
        --dtype fp32 \
	--eval_sample_size 10000 --collection_sample_size 100000 \
	--attn_frac 0.10 \
    	--ff_frac 0.00 \
	--ff_scoring random \
	--attn_scoring random \
	--wandb_project nicky-peaks-test \
	--recalculate_activations False \
	--name "$@"
}

prune_rocket() {
    # ViT CIFAR100
    poetry run python /root/separability/examples/prune_30.py Ahmed9275/Vit-Cifar100 \
	--focus cifar100-rocketless --cripple cifar100-rocket \
        --dtype fp32 \
	--eval_sample_size 1000 --collection_sample_size 100000 \
	--additional_datasets cifar100-rocket-mia \
        --wandb_project rockets \
	--recalculate_activations false \
	--name "$@" --n_steps 10
}

prune_mushrooms() {
    # ViT CIFAR100
    poetry run python /root/separability/examples/prune_30.py Ahmed9275/Vit-Cifar100 \
	--focus cifar100-mushroomless --cripple cifar100-mushroom \
        --dtype fp32 \
	--eval_sample_size 1000 --collection_sample_size 100000 \
	--additional_datasets cifar100-mushroom-mia \
        --wandb_project mushrooms \
	--recalculate_activations false \
	--name "$@"
}

nicky_prune_roberta() {
    #poetry run python /root/separability/examples/prune_30.py google/vit-base-patch16-224 \
    poetry run python prune_30.py roberta-large \
	--focus pile --cripple code \
        --dtype fp16 \
	--eval_sample_size 100000 --collection_sample_size 100000 \
	--attn_frac 0.05 \
    	--ff_frac 0.00 \
	--ff_scoring random \
	--attn_scoring random \
	--wandb_project nicky-peaks-test \
	--recalculate_activations False \
	--name "$@"
}

echo "Starting..."
# prune_vit "vit l 2% 2% noniter" --attn_frac 0.02 --ff_frac 0.02
# prune_vit "vit l 5% 0% random"  --attn_frac 0.02 --ff_frac 0.02 --attn_scoring random --ff_scoring random
#prune_mushrooms "vit b 3% 3% noniter" --attn_frac 0.03 --ff_frac 0.03 --n_steps 1
#prune_rocket "vit b 1% 1% noniter" --attn_frac 0.01 --ff_frac 0.01
#prune_vit_test "vit quick test"
#ben_prune_vit "vit revamped test"
#ben_prune_vit "vit revamped test"
#ben_prune_vit "vit revamped test"
nicky_prune_vit "roberta 5% peak" --attn_offset_mode "peak"
nicky_prune_vit "roberta 5% zero" --attn_offset_mode "zero"
nicky_prune_vit "roberta 5% mean" --attn_offset_mode "mean"
