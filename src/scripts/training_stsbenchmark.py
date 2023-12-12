# adapted from 

from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random
import torch
import numpy as np
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert-base-cased", type=str)
parser.add_argument("--model_path", default=None, type=str)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_seeds", default=5, type=int)
parser.add_argument("--pooling", default="mean", type=str) # cls, mean, max
args = parser.parse_args()


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)


model_save_path = model_save_path = 'output/training_stsbenchmark_'+args.model_name.replace("/", "-")
for seed in range(args.num_seeds):
    # Setting seed for all random initializations
    logging.info("##### Seed {} #####".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(args.model_name)
    if args.model_path is not None:
        word_embedding_model.auto_model = torch.load(args.model_path)

    # Apply mean pooling to get one fixed sized sentence vector
    if args.pooling == "cls":
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=True,
            pooling_mode_max_tokens=False
        )
    elif args.pooling == "mean":
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
    elif args.pooling == "max":
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=True
        )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    curr_model_save_path = model_save_path + '/seed_'+str(seed)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    logging.info("Read STSbenchmark dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs  * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=curr_model_save_path
    )

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    model = SentenceTransformer(curr_model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    test_evaluator(model, output_path=curr_model_save_path)


test_cosine_spearman = []
for seed in range(args.num_seeds):
    curr_model_save_path = model_save_path + '/seed_'+str(seed)+'/'

    # dev_results  = pd.read_csv(curr_model_save_path + '/eval/similarity_evaluation_sts-dev_results.csv')
    test_results = pd.read_csv(curr_model_save_path + 'similarity_evaluation_sts-test_results.csv')
    test_cosine_spearman.append(test_results["cosine_spearman"][0])

mean_cosine_spearman_test = np.mean(test_cosine_spearman)
stdev_cosine_spearman_test = np.std(test_cosine_spearman, ddof=1)

with open(model_save_path + '/test_results.json', 'w') as f:
    json.dump({
        "mean_cosine_spearman_test": mean_cosine_spearman_test,
        "stdev_cosine_spearman_test": stdev_cosine_spearman_test,
    }, f)