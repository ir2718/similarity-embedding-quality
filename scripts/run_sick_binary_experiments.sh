### SICK BINARY EXPERIMENTS ###

for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name microsoft/deberta-v3-base --dataset sick --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-base-discriminator --dataset sick --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-base-generator --dataset sick --starting_state $i --save_results
    python -m src.scripts.training --model_name bert-base-cased --dataset sick --starting_state $i --save_results
done
