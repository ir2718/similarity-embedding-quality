from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import logging
import os
import random
import torch
import numpy as np
import json
import argparse
from tqdm import tqdm
from copy import deepcopy
from src.scripts.hf_model import Model
from src.scripts.utils import *
from src.scripts.pooling_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="google/electra-base-discriminator", type=str)
parser.add_argument("--pooling_fn", default="mean", type=str) # cls, mean, weighted_mean, weighted_per_component_mean
parser.add_argument("--last_k_states", default=1, type=int)
parser.add_argument("--starting_state", default=12, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--test_batch_size", default=64, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--unsupervised", action="store_true")
parser.add_argument("--dataset", type=str, default="stsb") # stsb, kor_sts, serbian_sts
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_seeds", default=5, type=int)
parser.add_argument("--model_save_path", default="output", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

if args.last_k_states != 1 and args.pooling_fn not in ["mean", "weighted_mean", "cls", "max"]:
    raise Exception("Using last k hidden states is permitted with mean, weighted mean, cls and max pooling.")

if args.dataset == "stsb":
    loader_f = load_stsb
elif args.dataset == "kor_sts":
    loader_f = load_kor_sts
elif args.dataset == "serbian_sts":
    loader_f = load_sts_news_sr
elif args.dataset == "nli":
    loader_f = load_nli

train_loader, validation_loader, test_loader = loader_f(args.train_batch_size, args.test_batch_size)

model_dir = os.path.join(
    args.model_save_path, 
    args.model_name.replace('/', '-'),
    args.pooling_fn,
    f"{args.starting_state}_to_{args.starting_state + args.last_k_states}"
)
os.makedirs(model_dir, exist_ok=True)

if args.dataset in ["stsb", "kor_sts", "serbian_sts"]:
    loss_f = nn.MSELoss() 
elif args.dataset == "nli":
    loss_f = nn.CrossEntropyLoss()

test_cosine_spearman, test_cosine_pearson = [], []
if not args.unsupervised:
    for seed in range(args.num_seeds):
        logging.info("##### Seed {} #####".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        model = Model(args).to(args.device)

        optimizer_grouped_parameters = remove_params_from_optimizer(model, args.weight_decay)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
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
                
                loss = loss_f(out, score)
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

        test_pearson, test_spearman = model.validate(test_loader, args.device)
        test_cosine_pearson.append(test_pearson)
        test_cosine_spearman.append(test_spearman)

        print("============ TEST ============")
        print(f"Spearman - {test_spearman}")
        print(f" Pearson - {test_pearson}\n")

else:
    model = Model(args).to(args.device)

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

json_res_path = os.path.join(model_dir, "test_results" + (f"_args.dataset" if args.dataset != "stsb" else "") + ("_unsupervised" if args.unsupervised else "") + ".json")

with open(json_res_path, "w") as f:
    json.dump({
        "mean_cosine_spearman_test": mean_cosine_spearman_test,
        "stdev_cosine_spearman_test": stdev_cosine_spearman_test,
        "mean_cosine_pearson_test": mean_cosine_pearson_test,
        "stdev_cosine_pearson_test": stdev_cosine_pearson_test,
        "values_spearman": test_cosine_spearman,
        "values_pearson": test_cosine_pearson,
    }, f)