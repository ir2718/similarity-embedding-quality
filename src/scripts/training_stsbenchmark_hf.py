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
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
from src.scripts.hf_model import Model
from src.scripts.utils import *
from src.scripts.pooling_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="google/electra-base-discriminator", type=str)
parser.add_argument("--pooling_fn", default="mean", type=str) # cls, mean, weighted_mean, weighted_per_component_mean
parser.add_argument("--final_layer", default="cosine", type=str) # cosine, manhattan, euclidean, dot
parser.add_argument("--loss_function", default="mse", type=str) # mse, cross_entropy
parser.add_argument("--num_frozen_layers", default=0, type=int)
parser.add_argument("--starting_freeze", default=11, type=int)
parser.add_argument("--last_k_states", default=1, type=int)
parser.add_argument("--starting_state", default=12, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--test_batch_size", default=64, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--unsupervised", action="store_true")
parser.add_argument("--dataset", type=str, default="stsb") # stsb, kor_sts, serbian_sts
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_seeds", default=5, type=int)
parser.add_argument("--model_load_path", default=None, type=str)
parser.add_argument("--model_save_path", default="output", type=str)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

if args.last_k_states != 1 and args.pooling_fn not in ["mean", "weighted_mean", "cls", "max"]:
    raise Exception("Using last k hidden states is permitted with mean, weighted mean, cls and max pooling.")

if args.dataset == "stsb":
    loader_f = load_stsb
elif args.dataset == "kor_sts":
    loader_f = load_kor_sts
elif args.dataset == "spanish_sts":
    loader_f = load_spanish_sts
elif args.dataset == "german_sts":
    loader_f = load_german_sts
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

test_cosine_spearman, test_cosine_pearson = [], []
if not args.unsupervised:
    for seed in range(args.num_seeds):
        logging.info("##### Seed {} #####".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        model = Model(args).to(args.device)
        loss_dict = {
            "mse": nn.MSELoss(),
            "cross_entropy": nn.CrossEntropyLoss(),
        }
        loss_f = loss_dict[args.loss_function]

        if args.num_frozen_layers > 0:
            layers = model.model.encoder.layer if hasattr(model.model.encoder, "layer") else model.model.encoder.layers
            for l in layers[args.starting_freeze : args.starting_freeze + args.num_frozen_layers]:
                for p in l.parameters():
                    p.requires_grad = False

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
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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
    
        if args.save_model:
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y_%m_%d_%H_%M")
            torch.save(
                best_model.model, 
                os.path.join(model_dir, f"model_{formatted_time}.pkl")
            )

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

if not args.save_model:
    json_res_path = os.path.join(
        model_dir, 
        "test_results" + 
            (f"_{args.dataset}" if args.dataset != "stsb" else "") + 
            ("_unsupervised" if args.unsupervised else "") + 
            (f"_{args.final_layer}" if args.final_layer != "cosine" else "") + 
            (f"_frozen_{args.starting_freeze}_to_{args.starting_freeze + args.num_frozen_layers}" if args.num_frozen_layers != 0 else "") + 
            (f"_{'_'.join(args.model_load_path.split('/')[2].split('_')[1:])}" if not args.model_load_path is None else "") +
            ".json"
        )

    with open(json_res_path, "w") as f:
        json.dump({
            "mean_cosine_spearman_test": mean_cosine_spearman_test,
            "stdev_cosine_spearman_test": stdev_cosine_spearman_test,
            "mean_cosine_pearson_test": mean_cosine_pearson_test,
            "stdev_cosine_pearson_test": stdev_cosine_pearson_test,
            "values_spearman": test_cosine_spearman,
            "values_pearson": test_cosine_pearson,
        }, f)