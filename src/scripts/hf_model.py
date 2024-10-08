from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from src.scripts.pooling_functions import *
from src.scripts.utils import batch_to_device
from tqdm import tqdm
from src.scripts.final_layers import *
from safetensors.torch import load_file
from src.scripts.evaluators import STSEvaluator, SentencePairEvaluator, NLIEvaluator

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
        if args.random_init:
            self.model = AutoModel.from_config(self.config)

        final_layer_dict = {
            "cosine": CosineSimilarity(),
            "diff_concatenation": DifferenceConcatenation(
                hidden_size=self.config.hidden_size,
                num_classes=3,
            ),
        }
        self.final_layer = final_layer_dict[args.final_layer]
        
        if args.pooling_fn == "mean":
            self.pooling_fn = MeanPooling(args.starting_state)
        elif args.pooling_fn == "max":
            self.pooling_fn = MaxPooling(args.starting_state)
        elif args.pooling_fn == "cls":
            self.pooling_fn = CLSPooling(args.starting_state)

        self.dataset = args.dataset

        if self.dataset in ["sick"]:
            self.evaluator = NLIEvaluator()
        if self.dataset in ["mrpc"]:
            self.evaluator = NLIEvaluator()
            # self.evaluator = SentencePairEvaluator()
        elif self.dataset in ["stsb", "kor_sts", "spanish_sts", "german_sts"]:
            self.evaluator = STSEvaluator()

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

        scores_np = scores.detach().cpu().numpy()

        self.model.train()
        
        return self.evaluator(embeddings, scores_np)