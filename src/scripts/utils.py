from sentence_transformers import util
from torch.utils.data import DataLoader, Dataset
import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import gzip
import csv
import os

def read_data(path, dataset_f, train_batch_size, test_batch_size, dataset_kwargs={}, sample_f=lambda x: x):
    train_samples = []
    val_samples = []
    test_samples = []
    with gzip.open(path, 'rt', encoding='utf8') as f_in:
        mapping = {"contradiction": 0, "entailment": 1, "neutral": 2}
        reader = csv.DictReader(f_in, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:

            if "label" in row.keys():
                label = mapping[row["label"]]
            else:
                label = row["score"]

            inp_example = (row['sentence1'].strip(), row['sentence2'].strip(), label)

            if row['split'] == 'train':
                train_samples.append(inp_example)
            elif row['split'] == 'dev':
                val_samples.append(inp_example)
            else:
                test_samples.append(inp_example)

    train_samples = sample_f(train_samples)

    train_dataset = dataset_f(train_samples, **dataset_kwargs)
    val_dataset = dataset_f(val_samples, **dataset_kwargs)
    test_dataset = dataset_f(test_samples, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    return train_loader, val_loader, test_loader

class BaseDataset(Dataset):
    def __init__(self, samples, scale=None):
        # samples in format [[sentences1], [sentences2], [scores or labels]]
        # scores range is [0, 5], labels are strings
        self.s1, self.s2, self.scores = [], [], []
        for s1, s2, score in samples:
            self.s1.append(s1)
            self.s2.append(s2)
            if scale is None:
                self.scores.append(score)
            else:
                self.scores.append(float(score) / scale)

    def __getitem__(self, idx):
        return (self.s1[idx], self.s2[idx], torch.tensor(self.scores[idx]))

    def __len__(self):
        return len(self.s1)
    
def load_german_sts(train_batch_size, test_batch_size):
    data = datasets.load_dataset("stsb_multi_mt", "de")
    columns = ["sentence1", "sentence2", "score"]

    train = data["train"].to_pandas().rename(columns={'similarity_score': 'score'})[columns].values.tolist()
    val = data["dev"].to_pandas().rename(columns={'similarity_score': 'score'})[columns].values.tolist()
    test = data["test"].to_pandas().rename(columns={'similarity_score': 'score'})[columns].values.tolist()

    train_dataset = BaseDataset(train, scale=5.0)
    val_dataset = BaseDataset(val, scale=5.0)
    test_dataset = BaseDataset(test, scale=5.0)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    return train_loader, val_loader, test_loader

def load_spanish_sts(train_batch_size, test_batch_size):
    data = datasets.load_dataset("stsb_multi_mt", "es")
    columns = ["sentence1", "sentence2", "score"]

    train = data["train"].to_pandas().rename(columns={'similarity_score': 'score'})[columns].values.tolist()
    val = data["dev"].to_pandas().rename(columns={'similarity_score': 'score'})[columns].values.tolist()
    test = data["test"].to_pandas().rename(columns={'similarity_score': 'score'})[columns].values.tolist()

    train_dataset = BaseDataset(train, scale=5.0)
    val_dataset = BaseDataset(val, scale=5.0)
    test_dataset = BaseDataset(test, scale=5.0)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    return train_loader, val_loader, test_loader

def load_kor_sts(train_batch_size, test_batch_size):
    sts_dataset_path = 'datasets/kor-nlu-datasets/KorSTS/'
    
    columns = ["sentence1", "sentence2", "score"]

    train = pd.read_csv(os.path.join(sts_dataset_path, "sts-train.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE)[columns].values.tolist()
    val = pd.read_csv(os.path.join(sts_dataset_path, "sts-dev.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE)[columns].values.tolist()
    test = pd.read_csv(os.path.join(sts_dataset_path, "sts-test.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE)[columns].values.tolist()

    train_dataset = BaseDataset(train, scale=5.0)
    val_dataset = BaseDataset(val, scale=5.0)
    test_dataset = BaseDataset(test, scale=5.0)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    return train_loader, val_loader, test_loader

def load_stsb(train_batch_size, test_batch_size):
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
        
    return read_data(sts_dataset_path, BaseDataset, train_batch_size, test_batch_size, dataset_kwargs={"scale": 5.0})

def load_nli(train_batch_size, test_batch_size):
    nli_dataset_path = 'datasets/AllNLI.tsv.gz'

    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    return read_data(nli_dataset_path, BaseDataset, train_batch_size, test_batch_size)

def batch_to_device(x, device):
    return {k:v.to(device) for k, v in x.items()}

def remove_params_from_optimizer(model, weight_decay):
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
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters