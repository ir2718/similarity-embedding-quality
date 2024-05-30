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
from torch.nn.functional import cross_entropy
from src.scripts.hf_model import Model
from src.scripts.utils import *
from src.scripts.pooling_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="google/electra-base-discriminator", type=str)
parser.add_argument("--pooling_fn", default="mean", type=str) # cls, mean, max
parser.add_argument("--final_layer", default="cosine", type=str) 
parser.add_argument("--starting_state", default=12, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--test_batch_size", default=64, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--unsupervised", action="store_true")
parser.add_argument("--dataset", type=str, default="stsb") # stsb, mrpc, sst2
parser.add_argument("--best_metric_str", type=str, default="spearman") # spearman, map, mrr
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_seeds", default=5, type=int)
parser.add_argument("--model_load_path", default=None, type=str)
parser.add_argument("--model_save_path", default="output", type=str)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_results", action="store_true")
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

# sentence pair classification dataset
if args.dataset == "mrpc":
    loader_f = load_mrpc
    if args.best_metric_str == "spearman":
        args.best_metric_str = "map"

# retrieval dataset
elif args.dataset == "scifact":
    loader_f = load_scifact

# STS datasets
elif args.dataset == "stsb":
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

test_metrics = []
val_metrics = []
for seed in range(args.num_seeds):
    logging.info("##### Seed {} #####".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = Model(args).to(args.device)

    if args.dataset in ["mrpc"]:
        loss_f = nn.BCEWithLogitsLoss()
        pairwise = False
    elif args.dataset in ["stsb"]:
        loss_f = nn.MSELoss()
        pairwise = False
    elif args.dataset in ["scifact"]:
        def mnrl_loss(scores, labels):
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )  # Example a[i] should match with b[i]
            return cross_entropy(scores, labels)
        loss_f = mnrl_loss
        pairwise = True

    optimizer_grouped_parameters = remove_params_from_optimizer(model, args.weight_decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * args.num_epochs*len(train_loader)),
        num_training_steps=args.num_epochs*len(train_loader)
    )
    
    # training setup is same as in sentence transformers library
    best_epoch_idx, best_metric, best_model = 0, None, None
    for e in range(args.num_epochs):
        for *texts, score in tqdm(train_loader):
            tokenized_device = []
            for t in texts:
                tok_text = model.tokenizer(t, padding=True, truncation=True, return_tensors="pt")
                tok_text_device = batch_to_device(tok_text, args.device) 
                tokenized_device.append(tok_text_device)
            score = score.to(args.device)

            out = model.forward(tokenized_device, pairwise)

            loss = loss_f(out, score)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        metrics = model.validate(validation_loader, args.device)

        if best_metric is None or metrics[args.best_metric_str] > best_metric:
            best_epoch_idx = e
            best_optimized_metric = metrics[args.best_metric_str]
            best_model = deepcopy(model.cpu())
            model.to(args.device)

        print("============ VALIDATION ============")
        log_eval_results(metrics, e, args.num_epochs)
    
    if args.save_model:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        torch.save(
            best_model.model, 
            os.path.join(model_dir, f"model_{formatted_time}_{args.dataset}.pt")
        )

    print(f"Best epoch idx: {best_epoch_idx}\nBest metric: {best_optimized_metric}")
    best_model = best_model.to(args.device)

    new_val_metric = model.validate(validation_loader, args.device)
    val_metrics.append(new_val_metric)

    new_test_metric = model.validate(test_loader, args.device)
    test_metrics.append(new_test_metric)

    print("============ TEST ============")
    log_eval_results(new_test_metric, best_epoch_idx, args.num_epochs)
    
if args.save_results:

    test_json = create_json_dict(test_metrics, args.final_layer, "test") 
    json_res_path_test = os.path.join(
        model_dir, 
        "test_results" + 
            (f"_{args.dataset}" if args.dataset != "stsb" else "") + 
            (f"_{args.final_layer}" if args.final_layer != "cosine" else "") + 
            f"_{args.dataset}.json"
        )
    with open(json_res_path_test, "w") as f:
        json.dump(test_json, f, indent=4)


    val_json = create_json_dict(val_metrics, args.final_layer, "val")
    json_res_path_val = os.path.join(
        model_dir, 
        "val_results" + 
            (f"_{args.dataset}" if args.dataset != "stsb" else "") + 
            (f"_{args.final_layer}" if args.final_layer != "cosine" else "") + 
            f"_{args.dataset}.json"
        )
    with open(json_res_path_val, "w") as f:
        json.dump(val_json, f, indent=4)