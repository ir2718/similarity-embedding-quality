### KORSTS EXPERIMENTS ###
for ((i=0;i<=24;i++)); 
do 
    python -m src.scripts.training --model_name klue/bert-base --dataset kor_sts --starting_state $i --save_results
    python -m src.scripts.training --model_name monologg/koelectra-base-v3-discriminator --dataset kor_sts --starting_state $i --save_results
    python -m src.scripts.training --model_name monologg/koelectra-base-v3-generator --dataset kor_sts --starting_state $i --save_results
done

### GERMAN STSB EXPERIMENTS ###
for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name google-bert/bert-base-german-cased --dataset german_sts --starting_state $i --save_results
    python -m src.scripts.training --model_name deepset/gelectra-base --dataset german_sts --starting_state $i --save_results
    python -m src.scripts.training --model_name deepset/gelectra-base-generator --dataset german_sts --starting_state $i --save_results
done

### SPANISH STSB EXPERIMENTS ###
for ((i=0;i<=12;i++)); 
do 
    python -m src.scripts.training --model_name dccuchile/bert-base-spanish-wwm-cased --dataset spanish_sts --starting_state $i --save_results
    python -m src.scripts.training --model_name mrm8488/electricidad-base-discriminator --dataset spanish_sts --starting_state $i --save_results
    python -m src.scripts.training --model_name mrm8488/electricidad-base-generator --dataset spanish_sts --starting_state $i --save_results
done