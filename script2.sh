for ((i=7;i<=12;i++)); do
    python -m src.scripts.training_stsbenchmark_hf --model_name google/electra-base-discriminator --model_load_path ./output/google-electra-base-discriminator_word_similarity_dataset/seed_0/pytorch_model.bin --num_seeds 1 --starting_state $i --pooling_fn mean
done
