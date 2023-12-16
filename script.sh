for ((i=0;i<=12;i++)); do
    python -m src.scripts.training_stsbenchmark_hf --num_seeds 1 --model_name google/electra-base-generator --starting_state $i --pooling_fn mean_encoder
done


