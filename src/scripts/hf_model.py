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
        else:
            self.final_layer = self.final_layer()

        self.dataset = args.dataset
        
        if args.pooling_fn == "mean":
            self.pooling_fn = MeanPooling(args.starting_state)
        elif args.pooling_fn == "max":
            self.pooling_fn = MaxPooling(args.starting_state)
        elif args.pooling_fn == "cls":
            self.pooling_fn = CLSPooling(args.starting_state)
            
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
    def validate(self, loader, device, threshold=None):
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

        scores_np = scores.detach().cpu().numpy()

        self.model.train()
        
        if self.dataset in ["mrpc", "sst2"]:
            embeddings_np = embeddings.sigmoid().detach().cpu().numpy()
            scores_np = scores_np.astype(np.int32)

            thresholds = [k*0.01 for k in range(1, 100)]

            best_f1, acc, recall, prec = None, None, None, None
            
            if threshold is None:

                # find best threshold
                for t in thresholds:
                    preds = (embeddings_np > t).astype(np.int32)

                    acc_val = accuracy_score(scores_np, preds)
                    f1_val = f1_score(scores_np, preds)
                    recall_val = recall_score(scores_np, preds, zero_division=0.0)
                    precision_val = precision_score(scores_np, preds, zero_division=0.0)

                    if best_f1 is None or f1_val > best_f1:
                        best_threshold = t
                        best_f1 = f1_val
                        acc = acc_val
                        recall = recall_val
                        prec = precision_val

                print("Best threshold:",best_threshold)
                return acc, best_f1, recall, prec, best_threshold
        
            preds = (embeddings_np > threshold).astype(np.int32)
            acc = accuracy_score(scores_np, preds)
            f1 = f1_score(scores_np, preds)
            recall = recall_score(scores_np, preds, zero_division=0.0)
            prec = precision_score(scores_np, preds, zero_division=0.0)

            return acc, f1, recall, prec

        else:
            embeddings_np = embeddings.detach().cpu().numpy()
            return pearsonr(embeddings_np, scores_np)[0], spearmanr(embeddings_np, scores_np)[0]