import os
import numpy as np
import json
import argparse
import torch
import random
from src.scripts.hf_model import Model
from src.scripts.utils import *
from src.scripts.pooling_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="google/electra-base-discriminator", type=str)
parser.add_argument("--pooling_fn", default="mean", type=str) # cls, mean, weighted_mean, weighted_per_component_mean
parser.add_argument("--final_layer", default="cosine", type=str) # cosine, manhattan, euclidean, dot
parser.add_argument("--last_k_states", default=1, type=int)
parser.add_argument("--starting_state", default=12, type=int)
parser.add_argument("--test_batch_size", default=64, type=int)
parser.add_argument("--model_save_path", default="output", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

loader_f = load_stsb

_, validation_loader, _ = loader_f(args.test_batch_size, args.test_batch_size)

model_dir = os.path.join(
    args.model_save_path, 
    args.model_name.replace('/', '-'),
    args.pooling_fn,
    f"{args.starting_state}_to_{args.starting_state + args.last_k_states}"
)
print(f"================================= USING STATE {args.starting_state} =================================")
os.makedirs(model_dir, exist_ok=True)

val_cosine_spearman, val_cosine_pearson = [], []

models = [os.path.join(model_dir, x) for x in os.listdir(model_dir) if x.endswith(".pt")]
for i, m in enumerate(models):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    print(m)
    args.model_load_path = m
    model = Model(args).to(args.device)

    val_pearson, val_spearman = model.validate(validation_loader, args.device)
    val_cosine_pearson.append(val_pearson)
    val_cosine_spearman.append(val_spearman)

    print("============ VALIDATION ============")
    print(f"Spearman - {val_spearman}")
    print(f" Pearson - {val_pearson}\n")

mean_cosine_spearman_val = np.mean(val_cosine_spearman)
stdev_cosine_spearman_val = np.std(val_cosine_spearman, ddof=1)

mean_cosine_pearson_val = np.mean(val_cosine_pearson)
stdev_cosine_pearson_val = np.std(val_cosine_pearson, ddof=1)

json_res_path = os.path.join(
    model_dir, 
    "val_results.json"
    )

with open(json_res_path, "w") as f:
    json.dump({
        "mean_cosine_spearman_val": mean_cosine_spearman_val,
        "stdev_cosine_spearman_val": stdev_cosine_spearman_val,
        "mean_cosine_pearson_val": mean_cosine_pearson_val,
        "stdev_cosine_pearson_val": stdev_cosine_pearson_val,
        "values_spearman": val_cosine_spearman,
        "values_pearson": val_cosine_pearson,
    }, f)