for ((i=0;i<=13;i++)); do
    python -m src.scripts.training_stsbenchmark_hf --num_seeds 1 --model_name google/electra-base-discriminator --starting_state $i --pooling_fn mean_self_attention
done