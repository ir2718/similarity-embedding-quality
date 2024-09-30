### DAPT EXPERIMENTS ###

python -m src.scripts.dapt --model_name microsoft/deberta-v3-base --train_batch_size 8 --grad_accumulation_steps 32
python -m src.scripts.dapt --model_name google/electra-base-discriminator
python -m src.scripts.dapt --model_name google/electra-base-generator
python -m src.scripts.dapt --model_name bert-base-cased
