### MRPC EXPERIMENTS ###

for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name microsoft/deberta-v3-base --dataset mrpc --starting_state $i --save_results
    python -m src.scripts.training --model_name google/electra-base-discriminator --dataset mrpc --save_results --starting_state $i
    python -m src.scripts.training --model_name google/electra-base-generator --dataset mrpc --save_results --starting_state $i
    python -m src.scripts.training --model_name bert-base-cased --dataset mrpc --save_results --starting_state $i
done

