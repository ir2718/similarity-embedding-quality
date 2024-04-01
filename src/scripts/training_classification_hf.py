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
parser.add_argument("--final_layer", default="final_linear", type=str) 
parser.add_argument("--last_k_states", default=1, type=int)
parser.add_argument("--starting_state", default=12, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--test_batch_size", default=64, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--unsupervised", action="store_true")
parser.add_argument("--dataset", type=str, default="sst5") # sst5
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_seeds", default=5, type=int)
parser.add_argument("--model_load_path", default=None, type=str)
parser.add_argument("--model_save_path", default="output", type=str)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_results", action="store_true")
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

if args.last_k_states != 1 and args.pooling_fn not in ["mean", "weighted_mean", "cls", "max"]:
    raise Exception("Using last k hidden states is permitted with mean, weighted mean, cls and max pooling.")

if args.dataset == "sst5":
    loader_f = load_sst5
    args.num_classes = 5
else:
    raise Exception("Dataset is not available for usage with this script.")

train_loader, validation_loader, test_loader = loader_f(args.train_batch_size, args.test_batch_size)

model_dir = os.path.join(
    args.model_save_path, 
    args.model_name.replace('/', '-'),
    args.pooling_fn,
    f"{args.starting_state}_to_{args.starting_state + args.last_k_states}"
)
os.makedirs(model_dir, exist_ok=True)

test_acc, test_f1, test_recall, test_precision = [], [], [], []
val_acc, val_f1, val_recall, val_precision = [], [], [], []
for seed in range(args.num_seeds):
    logging.info("##### Seed {} #####".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = Model(args).to(args.device)
    loss_f = nn.CrossEntropyLoss()

    optimizer_grouped_parameters = remove_params_from_optimizer(model, args.weight_decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * args.num_epochs*len(train_loader)),
        num_training_steps=args.num_epochs*len(train_loader)
    )
    
    # training setup is same as in sentence transformers library
    for e in range(args.num_epochs):
        best_epoch_idx, best_acc, best_model = e, None, None
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

        acc, f1, recall, precision = model.validate(validation_loader, args.device)
        if best_acc is None or acc > best_acc:
            best_epoch_idx = e
            best_optimized_metric = acc
            best_model = deepcopy(model.cpu())
            model.to(args.device)

        print("============ VALIDATION ============")
        print(f"Epoch {e+1}/{args.num_epochs}")
        print(f" Accuracy - {acc}")
        print(f" F1 score - {f1}")
        print(f"   Recall - {recall}")
        print(f"Precision - {precision}\n")
    
    if args.save_model:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        torch.save(
            best_model.model, 
            os.path.join(model_dir, f"model_{formatted_time}_{args.dataset}.pt")
        )

    best_model = best_model.to(args.device)

    test_acc_val, test_f1_val, test_recall_val, test_precision_val = model.validate(test_loader, args.device)
    test_acc.append(test_acc_val)
    test_f1.append(test_f1_val)
    test_recall.append(test_recall_val)
    test_precision.append(test_precision_val)

    print("============ TEST ============")
    print(f" Accuracy - {test_acc_val}")
    print(f" F1 score - {test_f1_val}")
    print(f"   Recall - {test_recall_val}")
    print(f"Precision - {test_precision_val}\n")

    val_acc_val, val_f1_val, val_recall_val, val_precision_val  = model.validate(validation_loader, args.device)
    val_acc.append(val_acc_val)
    val_f1.append(val_f1_val)
    val_recall.append(val_recall_val)
    val_precision.append(val_precision_val)
    
if args.save_results:

    mean_acc_test = np.mean(test_acc)
    stdev_acc_test = np.std(test_acc, ddof=1)

    mean_f1_test = np.mean(test_f1)
    stdev_f1_test = np.std(test_f1, ddof=1)

    mean_recall_test = np.mean(test_recall)
    stdev_recall_test = np.std(test_recall, ddof=1)

    mean_precision_test = np.mean(test_precision)
    stdev_precision_test = np.std(test_precision, ddof=1)

    mean_acc_val = np.mean(val_acc)
    stdev_acc_val = np.std(val_acc, ddof=1)

    mean_f1_val = np.mean(val_f1)
    stdev_f1_val = np.std(val_f1, ddof=1)

    mean_recall_val = np.mean(val_recall)
    stdev_recall_val = np.std(val_recall, ddof=1)

    mean_precision_val = np.mean(val_precision)
    stdev_precision_val = np.std(val_precision, ddof=1)

    json_res_path_test = os.path.join(
        model_dir, 
        "test_results_" + 
            (f"_{args.dataset}" if args.dataset != "stsb" else "") + 
            (f"_{args.final_layer}" if args.final_layer != "cosine" else "") + 
            f"_{args.dataset}.json"
        )

    with open(json_res_path_test, "w") as f:
        json.dump({
            "mean_acc_test": mean_acc_test,
            "stdev_acc_test": stdev_acc_test,
            "mean_f1_test": mean_f1_test,
            "stdev_f1_test": stdev_f1_test,
            "mean_recall_test": mean_recall_test,
            "stdev_recall_test": stdev_recall_test,
            "mean_precision_test": mean_precision_test,
            "stdev_precision_test": stdev_precision_test,
            "values_acc": test_acc,
            "values_f1": test_f1,
            "values_recall": test_recall,
            "values_precision": test_precision,
        }, f)

    
    json_res_path_val = os.path.join(
        model_dir, 
        "val_results_" + 
            (f"_{args.dataset}" if args.dataset != "stsb" else "") + 
            (f"_{args.final_layer}" if args.final_layer != "cosine" else "") + 
            f"_{args.dataset}.json"
        )

    with open(json_res_path_val, "w") as f:
        json.dump({
            "mean_acc_val": mean_acc_val,
            "stdev_acc_val": stdev_acc_val,
            "mean_f1_val": mean_f1_val,
            "stdev_f1_val": stdev_f1_val,
            "mean_recall_val": mean_recall_val,
            "stdev_recall_val": stdev_recall_val,
            "mean_precision_val": mean_precision_val,
            "stdev_precision_val": stdev_precision_val,
            "values_acc": val_acc,
            "values_f1": val_f1,
            "values_recall": val_recall,
            "values_precision": val_precision,
        }, f)