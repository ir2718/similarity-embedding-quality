from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from src.scripts.pooling_functions import *
from src.scripts.utils import batch_to_device
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from src.scripts.final_layers import *
    
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModel.from_pretrained(args.model_name)
        if not args.model_load_path is None:
            self.model.load_state_dict(torch.load(args.model_load_path), strict=False)
        self.config = AutoConfig.from_pretrained(args.model_name)

        final_layer_dict = {
            "cosine": CosineSimilarity,
            "manhattan": ManhattanSimilarity,
            "euclidean": EuclideanSimilarity,
            "dot": DotProductSimilarity,
        }
        self.final_layer = final_layer_dict[args.final_layer]()
        
        if args.pooling_fn == "mean":
            self.pooling_fn = MeanPooling(args.last_k_states, args.starting_state)
        if args.pooling_fn == "gem":
            self.pooling_fn = GeMPooling(args.last_k_states, args.starting_state)
        if args.pooling_fn == "max_mean":
            self.pooling_fn = MaxMeanPooling(args.last_k_states, args.starting_state)
        if args.pooling_fn == "norm_mean":
            self.pooling_fn = NormMeanPooling(args.last_k_states, args.starting_state)
        elif args.pooling_fn == "mean_self_attention":
            self.pooling_fn = MeanSelfAttentionPooling(args.starting_state)
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
        x = self.tokenizer(s1, padding=True, truncation=True, return_tensors="pt")
        x = batch_to_device(x, self.model.device)
        return self.forward_once(x)

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