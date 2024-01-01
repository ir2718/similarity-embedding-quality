from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from src.scripts.pooling_functions import *
from src.scripts.utils import batch_to_device
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from src.scripts.final_layers import *
    
class DoubleModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        # tokenizer is the same
        self.combination = args.combination
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_disc)

        self.model_name_disc = args.model_name_disc
        self.model_disc = AutoModel.from_pretrained(args.model_name_disc)
        self.config_disc = AutoConfig.from_pretrained(args.model_name_disc)

        self.model_name_gen = args.model_name_gen
        self.model_gen = AutoModel.from_pretrained(args.model_name_gen)
        self.config_gen = AutoConfig.from_pretrained(args.model_name_gen)
        
        if args.pooling_fn == "mean":
            self.pooling_fn = MeanPooling(args.last_k_states, args.starting_state)
        if args.pooling_fn == "norm_mean":
            self.pooling_fn = NormMeanPooling(args.last_k_states, args.starting_state)
        elif args.pooling_fn == "mean_self_attention":
            self.pooling_fn = MeanSelfAttentionPooling(args.starting_state)
        elif args.pooling_fn == "max":
            self.pooling_fn = MaxPooling(args.last_k_states, args.starting_state)
        elif args.pooling_fn == "cls":
            self.pooling_fn = CLSPooling(args.last_k_states, args.starting_state)

        if "sts" in args.dataset:
            self.final_layer = CosineSimilarity()

    def forward_once(self, input):
        out_disc = self.model_disc(**input, output_hidden_states=True)
        out_gen = self.model_gen(**input, output_hidden_states=True)
        
        if self.combination == "concat":
            out = out_disc
            new_hidden_states = []
            for i in range(len(out["hidden_states"])):
                new_hidden_states.append(
                    torch.cat((out_disc["hidden_states"][i], out_gen["hidden_states"][i]), dim=-1)
                )
            out["hidden_states"] = tuple(new_hidden_states)
            out["last_hidden_state"] = new_hidden_states[-1]

        out_mean = self.pooling_fn(out, input["attention_mask"])
        return out_mean

    def forward(self, s1, s2):
        out1 = self.forward_once(s1)
        out2 = self.forward_once(s2)
        return self.final_layer(out1, out2)

    @torch.no_grad()
    def encode(self, s1):
        return self.forward_once(s1)

    @torch.no_grad()
    def validate(self, loader, device):
        embeddings = torch.tensor([]).to(device)
        scores = torch.tensor([]).to(device)
        for s1, s2, score in tqdm(loader):
            s1_tok = self.tokenizer(s1, padding=True, truncation=True, return_tensors="pt")
            s2_tok = self.tokenizer(s2, padding=True, truncation=True, return_tensors="pt")

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