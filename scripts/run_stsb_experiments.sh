### STSB EXPERIMENTS ###

for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name microsoft/deberta-v3-base --starting_state $i --save_results --save_model
    python -m src.scripts.training --model_name google/electra-base-discriminator --starting_state $i --save_results --save_model
    python -m src.scripts.training --model_name google/electra-base-generator --starting_state $i --save_results --save_model
    python -m src.scripts.training --model_name bert-base-cased --starting_state $i --save_results --save_model
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