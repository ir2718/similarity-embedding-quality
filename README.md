# Are ELECTRA's Sentence Embeddings Beyond Repair? The Case of Semantic Textual Similarity

Code for the paper [Are ELECTRA's Sentence Embeddings Beyond Repair? The Case of Semantic Textual Similarity](https://arxiv.org/abs/2402.13130) accepted at EMNLP 2024 Findings.

# Usage

```
git clone git@github.com:ir2718/similarity-embedding-quality.git
cd similarity-embedding-quality

python3 -m venv similarity_venv
source similarity_venv/bin/activate
pip3 install -r requirements.txt
```

# Reproducing Results

To reproduce the paper results, scripts are provided in the `scripts` folder:

```
chmod +x scripts/get_data.sh
chmod +x scripts/run_dapt.sh
chmod +x scripts/run_mrpc_experiments.sh
chmod +x scripts/run_random_stsb_experiments.sh
chmod +x scripts/run_sick_multiclass_experiments.sh
chmod +x scripts/run_stsb_experiments.sh
chmod +x scripts/run_stsb_improvements.sh
chmod +x scripts/run_translated_stsb_experiments.sh
chmod +x scripts/run_wordsim.sh
```

Get the word similarity data and the Korean dataset:
```
./scripts/get_data.sh
```

## STSB Various sizes
For reproducing the results on STSB for various model sizes:
```
./scripts/run_stsb_experiments.sh
```

## STSB With Improvements
For reproducing the results on STSB with improvements:
```
python3 src/scripts/preprocess_word_sim_data.py
./scripts/run_dapt.sh
./scripts/run_wordsim.sh
./scripts/run_stsb_improvements.sh
```

## Translated STSB
For reproducing the results in Korean, German, and Spanish:
```
./scripts/run_translated_stsb_experiments.sh
```

## MRPC
For reproducing the results on MRPC:
```
./scripts/run_mrpc_experiments.sh
```

## SICK
For reproducing the results on SICK:
```
./scripts/run_sick_multiclass_experiments.sh
```




