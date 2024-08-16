### RANDOMLY INITIALIZED MODEL STSB EXPERIMENTS ###

for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name microsoft/deberta-v3-base --starting_state $i --save_results --random_init
    python -m src.scripts.training --model_name google/electra-base-discriminator --starting_state $i --save_results --random_init
    python -m src.scripts.training --model_name google/electra-base-generator --starting_state $i --save_results --random_init
    python -m src.scripts.training --model_name bert-base-cased --starting_state $i --save_results --random_init
done