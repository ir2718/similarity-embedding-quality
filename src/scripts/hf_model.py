from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from src.scripts.pooling_functions import *
from src.scripts.utils import batch_to_device
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from src.scripts.final_layers import *
from safetensors.torch import load_file
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import numpy as np

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModel.from_pretrained(args.model_name)
        if not args.model_load_path is None:
            try:
                print("Trying to load state dict . . .")
                if args.model_load_path.endswith("model.safetensors"):
                    self.model.load_state_dict(load_file(args.model_load_path), strict=False)
                else:
                    self.model.load_state_dict(torch.load(args.model_load_path), strict=False)
            except TypeError:
                try:
                    print("Loading model instead . . .")
                    if args.model_load_path.endswith("model.safetensors"):
                        self.model = load_file(args.model_load_path)
                    else:
                        self.model = torch.load(args.model_load_path)
                except:
                    pass

        self.config = AutoConfig.from_pretrained(args.model_name)

        final_layer_dict = {
            "cosine": CosineSimilarity,
            "manhattan": ManhattanSimilarity,
            "euclidean": EuclideanSimilarity,
            "dot": DotProductSimilarity,
            "final_linear": FinalLinear,
            "diff_concatenation": DifferenceConcatenation
        }
        self.final_layer = final_layer_dict[args.final_layer]
        if args.final_layer in ["final_linear", "diff_concatenation"]:
            self.final_layer = self.final_layer(
                self.model.config.hidden_size, 
                args.num_classes-1 if args.num_classes == 2 else args.num_classes
            )

        self.dataset = args.dataset
        
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

    def forward(self, text):
        outs = [self.forward_once(x) for x in text]
        final_out = self.final_layer(*outs)
        if len(final_out.shape) == 2 and final_out.shape[1] == 1:
            return final_out.reshape(-1)
        return final_out

    @torch.no_grad()
    def encode(self, s1):
        x = self.tokenizer(s1, padding=True, truncation=True, return_tensors="pt")
        x = batch_to_device(x, self.model.device)
        return self.forward_once(x)

    @torch.no_grad()
    def validate(self, loader, device):
        self.model.eval()
        
        embeddings = torch.tensor([]).to(device)
        scores = torch.tensor([]).to(device)
        for *texts, score in tqdm(loader):
            tokenized_device = []
            for t in texts:
                tok_text = self.tokenizer(t, padding=True, truncation=True, return_tensors="pt")
                tok_text_device = batch_to_device(tok_text, device) 
                tokenized_device.append(tok_text_device)
            score = score.to(device)
            out = self.forward(tokenized_device)
            embeddings = torch.cat((embeddings, out), axis=0)
            scores = torch.cat((scores, score), axis=0)

        embeddings_np = embeddings.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        self.model.train()

        if self.dataset in ["sst5"]:
            preds = embeddings_np.argmax(axis=1).astype(np.int32)
            scores_np = scores_np.astype(np.int32)

            acc = accuracy_score(preds, scores_np)
            f1 = f1_score(preds, scores_np, average="macro")
            recall = recall_score(preds, scores_np, average="macro")
            precision = precision_score(preds, scores_np, average="macro")
            return acc, f1, recall, precision
        
        elif self.dataset in ["mrpc"]:
            preds = (embeddings_np > 0).astype(np.int32)
            scores_np = scores_np.astype(np.int32)
            
            print(preds.shape, scores_np.shape)
            print(preds)
            print(scores_np)

            acc = accuracy_score(preds, scores_np)
            f1 = f1_score(preds, scores_np)
            recall = recall_score(preds, scores_np)
            precision = precision_score(preds, scores_np)
            return acc, f1, recall, precision
        
        else:
            return pearsonr(embeddings_np, scores_np)[0], spearmanr(embeddings_np, scores_np)[0]