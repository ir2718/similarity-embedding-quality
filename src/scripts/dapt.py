from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.models.electra.modeling_electra import ElectraGeneratorPredictions
import torch.nn.functional as F
import torch.nn as nn
import os
import csv
import torch
import argparse
from tqdm import tqdm
from src.scripts.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="google/electra-base-discriminator", type=str)
parser.add_argument("--state_number", default=12, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--grad_accumulation_steps", default=8, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--model_save_path", default="dapt", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

###########################################################

model = AutoModel.from_pretrained(args.model_name).to(args.device)

if args.model_name == "google/electra-base-generator":
    generator_predictions = ElectraGeneratorPredictions(model.config)
    generator_lm_head = nn.Linear(model.config.embedding_size, model.config.vocab_size)
    generator_lm_head.weight = model.embeddings.word_embeddings.weight
    lm_head = nn.Sequential(generator_predictions, generator_lm_head).to(args.device)
else:
    lm_head = BertLMPredictionHead(model.config).to(args.device)
    lm_head.decoder.weight = model.embeddings.word_embeddings.weight

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

###########################################################

datasets_folder = "datasets"
os.makedirs(datasets_folder, exist_ok=True)
nli_dataset_path = os.path.join(datasets_folder, "AllNLI.tsv.gz")

if not os.path.exists(nli_dataset_path):
    util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

model_save_path = os.path.join(args.model_save_path, args.model_name.replace("/", "-"))
os.makedirs(model_save_path, exist_ok=True)

train_samples_set = set()
with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        train_samples_set.add(row["sentence1"])
        train_samples_set.add(row["sentence2"])
train_samples = list(train_samples_set)

############################################################
        
class DAPTDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)

dataset = DAPTDataset(train_samples)
dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

optimizer_grouped_parameters = remove_params_from_optimizer(model, args.weight_decay)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * args.num_epochs*len(dataloader)),
    num_training_steps=args.num_epochs*len(dataloader)
)

############################################################

iter_num = 1
track_losses = []

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    return_tensors="pt"
)

for e in range(args.num_epochs):
    batch_idx = 0
    for batch in tqdm(dataloader):
        tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs, labels = data_collator.torch_mask_tokens(tokenized["input_ids"])
        labels = labels.to(args.device)
        tokenized_gpu = batch_to_device(tokenized, args.device)
        out = model(**tokenized_gpu, output_hidden_states=True).hidden_states[args.state_number] # N, seq_len, hidden_size

        # labels are N, seq_len
        lm_out = lm_head(out) # N, seq_len, vocab_size

        loss = F.cross_entropy(lm_out.view(-1, tokenizer.vocab_size), labels.view(-1), ignore_index=-100)
            
        loss = loss / args.grad_accumulation_steps
        loss.backward()
        track_losses.append(loss.clone().detach() * args.grad_accumulation_steps)

        if iter_num % 100 == 0:
            print(f"Iteration {iter_num}: {loss.item()  * args.grad_accumulation_steps}")
            with open(os.path.join(model_save_path, f"losses.txt"), "a") as f:
                f.write("\n".join([str(x.cpu().tolist()) for x in track_losses]) + "\n")
            track_losses = []

        if ((batch_idx + 1) % args.grad_accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        iter_num += 1
        batch_idx += 1
    
    torch.save(
        model, 
        os.path.join(model_save_path, f"model_epoch_{e}_mlm.pt")
    )
    torch.save(
        lm_head, 
        os.path.join(model_save_path, f"head_epoch_{e}_mlm.pt")
    )