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
parser.add_argument("--pooling_fn", default="mean", type=str) # cls, mean, max
parser.add_argument("--final_layer", default="cosine", type=str) # cosine, manhattan, euclidean, dot
parser.add_argument("--starting_state", default=12, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--test_batch_size", default=64, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--unsupervised", action="store_true")
# semantic similarity - stsb, kor_sts, spanish_sts, german_sts
parser.add_argument("--dataset", type=str, default="stsb") 
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_seeds", default=5, type=int)
parser.add_argument("--model_load_path", default=None, type=str)
parser.add_argument("--model_save_path", default="output", type=str)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_results", action="store_true")
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

if args.dataset == "stsb":
    loader_f = load_stsb
elif args.dataset == "kor_sts":
    loader_f = load_kor_sts
elif args.dataset == "spanish_sts":
    loader_f = load_spanish_sts
elif args.dataset == "german_sts":
    loader_f = load_german_sts
else:
    raise Exception("Dataset is not available for usage with this script.")

train_loader, validation_loader, test_loader = loader_f(args.train_batch_size, args.test_batch_size)

model_dir = os.path.join(
    args.model_save_path, 
    args.model_name.replace('/', '-'),
    args.pooling_fn,
    f"{args.starting_state}_to_{args.starting_state + 1}"
)
os.makedirs(model_dir, exist_ok=True)

test_cosine_spearman, test_cosine_pearson = [], []
val_cosine_spearman, val_cosine_pearson = [], []
for seed in range(args.num_seeds):
    logging.info("##### Seed {} #####".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = Model(args).to(args.device)
    loss_f = nn.MSELoss()

    optimizer_grouped_parameters = remove_params_from_optimizer(model, args.weight_decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * args.num_epochs*len(train_loader)),
        num_training_steps=args.num_epochs*len(train_loader)
    )
    
    # training setup is same as in sentence transformers library
    best_epoch_idx, best_spearman, best_model = 0, None, None
    for e in range(args.num_epochs):
        for *texts, score in tqdm(train_loader):
            tokenized_device = []
            for t in texts:
                tok_text = model.tokenizer(t, padding=True, truncation=True, return_tensors="pt")
                tok_text_device = batch_to_device(tok_text, args.device) 
                tokenized_device.append(tok_text_device)
            score = score.to(args.device)

            out = model.forward(tokenized_device)
            
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
        formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        torch.save(
            best_model.model, 
            os.path.join(model_dir, f"model_{formatted_time}.pt")
        )

    print(f"Best epoch idx: {best_epoch_idx}")
    best_model = best_model.to(args.device)

    test_pearson, test_spearman = model.validate(test_loader, args.device)
    test_cosine_pearson.append(test_pearson)
    test_cosine_spearman.append(test_spearman)

    print("============ TEST ============")
    print(f"Spearman - {test_spearman}")
    print(f" Pearson - {test_pearson}\n")

    val_pearson, val_spearman = model.validate(validation_loader, args.device)
    val_cosine_pearson.append(val_pearson)
    val_cosine_spearman.append(val_spearman)

if args.save_results:

    mean_cosine_spearman_test = np.mean(test_cosine_spearman)
    stdev_cosine_spearman_test = np.std(test_cosine_spearman, ddof=1)

    mean_cosine_pearson_test = np.mean(test_cosine_pearson)
    stdev_cosine_pearson_test = np.std(test_cosine_pearson, ddof=1)

    mean_cosine_spearman_val = np.mean(val_cosine_spearman)
    stdev_cosine_spearman_val = np.std(val_cosine_spearman, ddof=1)

    mean_cosine_pearson_val = np.mean(val_cosine_pearson)
    stdev_cosine_pearson_val = np.std(val_cosine_pearson, ddof=1)

    if args.model_load_path is not None:
        if "dapt" in args.model_load_path:
            path_to_add = f"_{'_'.join(args.model_load_path.split('/')[2:]).split('.')[0]}"
        else:
            path_to_add = "_".join(args.model_load_path.split('/')[-4].split("_")[1:])

    json_res_path_test = os.path.join(
        model_dir, 
        "test_results" + 
            (f"_{args.dataset}" if args.dataset != "stsb" else "") + 
            (f"_{args.final_layer}" if args.final_layer != "cosine" else "") + 
            (path_to_add if args.model_load_path is not None else "") +
            ".json"
        )

    with open(json_res_path_test, "w") as f:
        json.dump({
            "mean_cosine_spearman_test": mean_cosine_spearman_test,
            "stdev_cosine_spearman_test": stdev_cosine_spearman_test,
            "mean_cosine_pearson_test": mean_cosine_pearson_test,
            "stdev_cosine_pearson_test": stdev_cosine_pearson_test,
            "values_spearman": test_cosine_spearman,
            "values_pearson": test_cosine_pearson,
        }, f, indent=4)

    
    json_res_path_val = os.path.join(
        model_dir, 
        "val_results" + 
            (f"_{args.dataset}" if args.dataset != "stsb" else "") + 
            (f"_{args.final_layer}" if args.final_layer != "cosine" else "") + 
            (path_to_add if args.model_load_path is not None else "") +
            ".json"
        )

    with open(json_res_path_val, "w") as f:
        json.dump({
            "mean_cosine_spearman_val": mean_cosine_spearman_val,
            "stdev_cosine_spearman_val": stdev_cosine_spearman_val,
            "mean_cosine_pearson_val": mean_cosine_pearson_val,
            "stdev_cosine_pearson_val": stdev_cosine_pearson_val,
            "values_spearman": val_cosine_spearman,
            "values_pearson": val_cosine_pearson,
        }, f, indent=4)