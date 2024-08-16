### STSB EXPERIMENTS ###


## WRITE SCRIPT FOR WORDSIM AND DAPT
## RUN DAPT AND WORDSIM FOR DEBERTA

## REDO ALL LARGE AND SMALL
## ADD XLNET 
## DAPT AND WORDSIM NEEDS ADDING DEBERTA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

for ((i=0;i<=12;i++)); 
do 
#     python -m src.scripts.training --model_name microsoft/deberta-v3-base --starting_state $i --save_results --save_model
#     python -m src.scripts.training --model_name google/electra-base-discriminator --starting_state $i --save_results --save_model
#     python -m src.scripts.training --model_name google/electra-base-generator --starting_state $i --save_results --save_model
#     python -m src.scripts.training --model_name bert-base-cased --starting_state $i --save_results --save_model
    python -m src.scripts.training --model_name xlnet/xlnet-base-cased --starting_state $i --save_results
done








# ### STSB DAPT EXPERIMENTS ###
for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name microsoft/deberta-v3-base --model_load_path ./dapt/microsoft-deberta-v3-base/model_epoch_9_mlm.pt --starting_state $i --save_results
#     python -m src.scripts.training --model_name google/electra-base-discriminator --model_load_path ./dapt/google-electra-base-discriminator/model_epoch_9_mlm.pt --starting_state $i --save_results
#     python -m src.scripts.training --model_name google/electra-base-generator --model_load_path ./dapt/google-electra-base-generator/model_epoch_9_mlm.pt --starting_state $i --save_results
#     python -m src.scripts.training --model_name bert-base-cased --model_load_path ./dapt/bert-base-cased/model_epoch_9_mlm.pt --starting_state $i --save_results
done

# ### STSB WORD SIMILARITY EXPERIMENTS ###
for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name microsoft/deberta-v3-base --model_load_path ./dapt/microsoft-deberta-v3-base_word_similarity/seed_0/$i/model.safetensors --starting_state $i --save_results
#     python -m src.scripts.training --model_name google/electra-base-discriminator --model_load_path ./output/google-electra-base-discriminator_word_similarity/seed_0/$i/model.safetensors --starting_state $i --save_results
#     python -m src.scripts.training --model_name google/electra-base-generator --model_load_path ./output/google-electra-base-generator_word_similarity/seed_0/$i/model.safetensors --starting_state $i --save_results
#     python -m src.scripts.training --model_name bert-base-cased --model_load_path ./output/bert-base-cased_word_similarity/seed_0/$i/model.safetensors --starting_state $i --save_results
done












### STSB LARGE EXPERIMENTS ###

for ((i=0;i<=24;i++)); 
do 
    python -m src.scripts.training --model_name google/electra-large-generator --starting_state $i --save_results
    python -m src.scripts.training --model_name bert-large-cased --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-large-discriminator --starting_state $i --save_results
done

### STSB SMALL EXPERIMENTS ###

for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name google/electra-small-generator --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-small-discriminator --starting_state $i --save_results
done

### STSB DIFFERENT BERT SIZE EXPERIMENTS ###

for ((i=0;i<=2;i++)); 
do 
    python -m src.scripts.training --model_name google/bert_uncased_L-2_H-128_A-2 --starting_state $i --save_results
done

for ((i=0;i<=4;i++)); 
do 
    python -m src.scripts.training --model_name google/bert_uncased_L-4_H-256_A-4 --starting_state $i --save_results
    python -m src.scripts.training --model_name google/bert_uncased_L-4_H-512_A-8 --starting_state $i --save_results
done

for ((i=0;i<=8;i++)); 
do 
    python -m src.scripts.training --model_name google/bert_uncased_L-8_H-512_A-8 --starting_state $i --save_results
done