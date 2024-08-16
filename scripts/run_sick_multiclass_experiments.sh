### SICK MULTICLASS EXPERIMENTS ###

for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name microsoft/deberta-v3-base --dataset sick --final_layer diff_concatenation --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-base-discriminator --dataset sick --final_layer diff_concatenation --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-base-generator --dataset sick --final_layer diff_concatenation --starting_state $i --save_results
    python -m src.scripts.training --model_name bert-base-cased --dataset sick --final_layer diff_concatenation --starting_state $i --save_results
done
