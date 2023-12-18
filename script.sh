for ((i=0;i<=13;i++)); do
    python -m src.scripts.training_stsbenchmark_hf --num_seeds 5 --model_name google/electra-base-generator --starting_state $i --pooling_fn cls
done