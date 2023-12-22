from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from src.scripts.pooling_functions import *
from src.scripts.utils import batch_to_device
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

class CosineSimilarity(nn.Module):

    def forward(self, out1, out2):
        out_1_norm = F.normalize(out1, p=2.0, dim=1)
        out_2_norm = F.normalize(out2, p=2.0, dim=1)
        return (out_1_norm * out_2_norm).sum(dim=1)

class DifferenceConcatenation(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 3, 3, bias=True)

    def forward(self, out1, out2):
        concatenation = torch.cat((out1, out2, torch.abs(out1 - out2)), dim=1)
        return self.linear(concatenation)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModel.from_pretrained(args.model_name)
        self.config = AutoConfig.from_pretrained(args.model_name)
        
        if args.pooling_fn == "mean":
            self.pooling_fn = MeanPooling(args.last_k_states, args.starting_state)
        elif args.pooling_fn == "mean_encoder":
            self.pooling_fn = MeanEncoderPooling(self.config, args.starting_state)
        elif args.pooling_fn == "max":
            self.pooling_fn = MaxPooling(args.last_k_states, args.starting_state)
        elif args.pooling_fn == "cls":
            self.pooling_fn = CLSPooling(args.last_k_states, args.starting_state)
        elif args.pooling_fn == "weighted_mean":
            self.pooling_fn = WeightedMeanPooling(self.config, args.last_k_states, args.starting_state)
        elif args.pooling_fn == "weighted_per_component_mean":
            self.pooling_fn = WeightedPerComponentMeanPooling(self.config, self.tokenizer)

        if args.dataset == "nli":
            self.final_layer = DifferenceConcatenation(self.config.hidden_size)
        elif args.dataset in ["stsb", "kor_sts", "serbian_sts"]:
            self.final_layer = CosineSimilarity()

    def forward_once(self, inputs):
        out = self.model(**inputs, output_hidden_states=True)
        out_mean = self.pooling_fn(out, inputs["attention_mask"])
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