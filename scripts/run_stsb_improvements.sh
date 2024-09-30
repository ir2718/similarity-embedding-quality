# ### STSB DAPT EXPERIMENTS ###
for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name microsoft/deberta-v3-base --model_load_path ./dapt/microsoft-deberta-v3-base/model_epoch_9_mlm.pt --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-base-discriminator --model_load_path ./dapt/google-electra-base-discriminator/model_epoch_9_mlm.pt --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-base-generator --model_load_path ./dapt/google-electra-base-generator/model_epoch_9_mlm.pt --starting_state $i --save_results
    python -m src.scripts.training --model_name bert-base-cased --model_load_path ./dapt/bert-base-cased/model_epoch_9_mlm.pt --starting_state $i --save_results
done

### STSB WORD SIMILARITY EXPERIMENTS ###
for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name microsoft/deberta-v3-base --model_load_path ./output/microsoft-deberta-v3-base_word_similarity/seed_0/$i/model.safetensors --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-base-discriminator --model_load_path ./output/google-electra-base-discriminator_word_similarity/seed_0/$i/model.safetensors --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-base-generator --model_load_path ./output/google-electra-base-generator_word_similarity/seed_0/$i/model.safetensors --starting_state $i --save_results
    python -m src.scripts.training --model_name bert-base-cased --model_load_path ./output/bert-base-cased_word_similarity/seed_0/$i/model.safetensors --starting_state $i --save_results
done
