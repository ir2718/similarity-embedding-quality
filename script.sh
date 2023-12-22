for ((i=0;i<=13;i++)); do
    python -m src.scripts.training_stsbenchmark_hf --num_seeds 5 --model_name kykim/electra-kor-base --starting_state $i --pooling_fn mean --dataset kor_sts
done
