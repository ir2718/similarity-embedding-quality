from typing import Any
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from sentence_transformers import util
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import os
import gzip
import csv
import random
import torch
import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import pearsonr, spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="google/electra-base-discriminator", type=str)
parser.add_argument("--pooling_fn", default="mean", type=str) # mean, weighted_mean, weighted_per_component_mean
parser.add_argument("--last_k_states", default=1, type=int)
parser.add_argument("--starting_state", default=12, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--test_batch_size", default=64, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_seeds", default=5, type=int)
parser.add_argument("--model_save_path", default="output", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

if args.last_k_states != 1 and args.pooling_fn not in ["mean", "weighted_mean"]:
    raise Exception("Using last k hidden states is only permitted with mean and weighted mean pooling.")

#########################################################################################
# loading data

sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = (row['sentence1'], row['sentence2'], score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

class STSB(Dataset):
    def __init__(self, samples):
        # samples in format [[sentences1], [sentences2], [scores]]
        self.s1, self.s2, self.scores = [], [], []
        for s1, s2, score in samples:
            self.s1.append(s1)
            self.s2.append(s2)
            self.scores.append(score)

    def __getitem__(self, idx):
        return (self.s1[idx], self.s2[idx], torch.tensor(self.scores[idx]))

    def __len__(self):
        return len(self.s1)

train_dataset = STSB(train_samples)
validation_dataset = STSB(dev_samples)
test_dataset = STSB(test_samples)

train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)
validation_loader = DataLoader(validation_dataset, batch_size=args.test_batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

#########################################################################################
# pooling types

class MeanCLSPooling(nn.Module):
    def __init__(self, last_k_states, starting_state):
        super().__init__()
        self.last_k = last_k_states
        self.starting_state = starting_state

    def forward(self, hidden, attention_mask):
        last_k_hidden = torch.stack(
            hidden.hidden_states[self.starting_state : self.starting_state + self.last_k]
        ).mean(dim=0, keepdim=True)[:, :, 0, :]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_k_hidden.size()).float()
        emb_sum = torch.sum(last_k_hidden * input_mask_expanded, dim=2)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=2), min=1e-9) # denominator
        emb_mean = emb_sum / sum_mask
        return emb_mean.squeeze(0)

class MeanPooling(nn.Module):
    def __init__(self, last_k_states, starting_state):
        super().__init__()
        self.last_k = last_k_states
        self.starting_state = starting_state

    def forward(self, hidden, attention_mask):
        last_k_hidden = torch.stack(
            hidden.hidden_states[self.starting_state : self.starting_state + self.last_k]
        ).mean(dim=0, keepdim=True)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_k_hidden.size()).float()
        emb_sum = torch.sum(last_k_hidden * input_mask_expanded, dim=2)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=2), min=1e-9) # denominator
        emb_mean = emb_sum / sum_mask
        return emb_mean.squeeze(0)

class WeightedMeanPooling(nn.Module):
    def __init__(self, config, last_k, starting_state):
        super().__init__()
        self.last_k = last_k
        self.starting_state = starting_state
        self.mean_weights = torch.nn.Parameter(torch.randn(config.hidden_size, self.last_k))
        self.mean_weights.requires_grad = True


    def forward(self, hidden, attention_mask):
        last_k_hidden = torch.stack(
            hidden.hidden_states[self.starting_state : self.starting_state + self.last_k]
        ).mean(dim=0, keepdim=True)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_k_hidden.size()).float()
        emb_sum = torch.sum(last_k_hidden * self.mean_weights * input_mask_expanded, dim=2)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=2), min=1e-9) # denominator
        emb_mean = emb_sum / sum_mask
        return emb_mean.squeeze(0)
    
class WeightedPerComponentMeanPooling(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.mean_weights = torch.nn.Parameter(torch.randn(tokenizer.model_max_length, config.hidden_size))
        self.mean_weights.requires_grad = True


    def forward(self, hidden, attention_mask):
        last_hidden = hidden.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        emb_sum = torch.sum(last_hidden * self.mean_weights[:last_hidden.shape[1]] * input_mask_expanded, dim=2)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9) # denominator
        emb_mean = emb_sum / sum_mask
        return emb_mean

#########################################################################################################
# model and train loop

class Model(nn.Module):
    def __init__(self, model_name, pooling_fn, last_k_states=1, starting_state=-1):
        super().__init__()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModel.from_pretrained(args.model_name)
        self.config = AutoConfig.from_pretrained(args.model_name)
        
        if pooling_fn == "mean":
            self.pooling_fn = MeanPooling(last_k_states, starting_state)
        elif pooling_fn == "weighted_mean":
            self.pooling_fn = WeightedMeanPooling(self.config, last_k_states, starting_state)
        elif pooling_fn == "weighted_per_component_mean":
            self.pooling_fn = WeightedPerComponentMeanPooling(self.config, self.tokenizer)

    def forward_once(self, inputs):
        out = self.model(**inputs, output_hidden_states=True)
        out_mean = self.pooling_fn(out, inputs["attention_mask"])
        out_emb_norm = F.normalize(out_mean, p=2.0, dim=1)
        return out_emb_norm

    def forward(self, s1, s2):
        out1 = self.forward_once(s1)
        out2 = self.forward_once(s2)

        return (out1 * out2).sum(dim=1)
    
    @torch.no_grad()
    def encode(self, s1):
        return self.forward_once(s1)

    @torch.no_grad()
    def validate(self, loader, device):
        embeddings = torch.tensor([]).to(device)
        scores = torch.tensor([]).to(device)
        for s1, s2, score in tqdm(loader):
            s1_tok = model.tokenizer(s1, padding=True, truncation=True, return_tensors="pt")
            s2_tok = model.tokenizer(s2, padding=True, truncation=True, return_tensors="pt")

            s1_tok_device = batch_to_device(s1_tok, device)
            s2_tok_device = batch_to_device(s2_tok, device)
            score = score.to(device)

            out = self.forward(s1_tok_device, s2_tok_device)

            embeddings = torch.cat((embeddings, out), axis=0)
            scores = torch.cat((scores, score), axis=0)

        embeddings_np = embeddings.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        pearson = pearsonr(embeddings_np, scores_np)[0]
        spearman = spearmanr(embeddings_np, scores_np)[0]

        return pearson, spearman

def batch_to_device(x, device):
    return {k:v.to(device) for k, v in x.items()}

#############################################################################################

model_dir = os.path.join(
    args.model_save_path, 
    f"{args.model_name.replace('/', '-')}_{args.pooling_fn}_{args.starting_state}_to_{args.starting_state + args.last_k_states}"
)
test_cosine_spearman, test_cosine_pearson = [], []

for seed in range(args.num_seeds):
    logging.info("##### Seed {} #####".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = Model(args.model_name, args.pooling_fn, args.last_k_states, args.starting_state).to(args.device)

    # taken from https://github.com/huggingface/transformers/blob/7c6cd0ac28f1b760ccb4d6e4761f13185d05d90b/src/transformers/trainer_pt_utils.py
    def get_parameter_names(model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * args.num_epochs*len(train_loader)),
        num_training_steps=args.num_epochs*len(train_loader)
    )
    
    # training setup is same as in sentence transformers library
    for e in range(args.num_epochs):
        best_epoch_idx, best_spearman, best_model = e, None, None
        for s1, s2, score in tqdm(train_loader):
            s1_tok = model.tokenizer(s1, padding=True, truncation=True, return_tensors="pt")
            s2_tok = model.tokenizer(s2, padding=True, truncation=True, return_tensors="pt")

            s1_tok_device = batch_to_device(s1_tok, args.device)
            s2_tok_device = batch_to_device(s2_tok, args.device)
            score = score.to(args.device)

            out = model.forward(s1_tok_device, s2_tok_device)
            
            loss = ((out - score)**2).mean()
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        val_pearson, val_spearman = model.validate(validation_loader, args.device)
        if best_spearman is None or val_spearman > best_spearman:
            best_epoch_idx = e
            best_spearman = val_spearman
            best_model = deepcopy(model.cpu())
            model.to(args.device)

        print("============ VALIDATION ============")
        print(f"Epoch {e+1}/{args.num_epochs}")
        print(f"Spearman - {val_spearman}")
        print(f" Pearson - {val_pearson}\n")
        

    best_model = best_model.to(args.device)
    curr_model_save_path = os.path.join(model_dir, f"seed_{str(seed)}")
    os.makedirs(curr_model_save_path, exist_ok=True)

    test_pearson, test_spearman = model.validate(test_loader, args.device)
    test_cosine_pearson.append(test_pearson)
    test_cosine_spearman.append(test_spearman)

    print("============ TEST ============")
    print(f"Spearman - {test_spearman}")
    print(f" Pearson - {test_pearson}\n")

mean_cosine_spearman_test = np.mean(test_cosine_spearman)
stdev_cosine_spearman_test = np.std(test_cosine_spearman, ddof=1)

mean_cosine_pearson_test = np.mean(test_cosine_pearson)
stdev_cosine_pearson_test = np.std(test_cosine_pearson, ddof=1)

json_res_path = os.path.join(model_dir, "test_results.json")
with open(json_res_path, "w") as f:
    json.dump({
        "mean_cosine_spearman_test": mean_cosine_spearman_test,
        "stdev_cosine_spearman_test": stdev_cosine_spearman_test,
        "mean_cosine_pearson_test": mean_cosine_pearson_test,
        "stdev_cosine_pearson_test": stdev_cosine_pearson_test,
    }, f)