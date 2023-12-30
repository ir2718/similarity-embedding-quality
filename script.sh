for ((i=0;i<=12;i++)); do
    python -m src.scripts.training_stsbenchmark_hf --num_seeds 1 --model_name bert-base-cased --starting_state $i --pooling_fn mean
done
