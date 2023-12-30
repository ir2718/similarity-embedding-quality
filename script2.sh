for ((i=0;i<=12;i++)); do
    python -m src.scripts.training_stsbenchmark_hf --num_seeds 5 --model_name deepset/gelectra-base --starting_state $i --pooling_fn mean --device cuda:1 --dataset german_sts
done
