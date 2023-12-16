for ((i=0;i<=12;i++)); do
    python -m src.scripts.training_stsbenchmark_hf --num_seeds 1 --model_name google/electra-base-discriminator --starting_state $i --pooling_fn max --device cuda:1
done