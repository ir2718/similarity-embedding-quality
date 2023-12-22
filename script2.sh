for ((i=0;i<=13;i++)); do
    python -m src.scripts.training_stsbenchmark_hf --num_seeds 1 --model_name EMBEDDIA/crosloengual-bert --starting_state $i --pooling_fn mean --dataset serbian_sts --device cuda:1
done
