'''
This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.

We do not mine hard negatives or train triplets in this example.

Running this script:
python train_sbert.py
'''

# adapted from https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_sbert.py

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="google/electra-base-discriminator", type=str)
parser.add_argument("--pooling_fn", default="mean", type=str)
parser.add_argument("--starting_state", default=12, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--test_batch_size", default=64, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--dataset", type=str, default="stsb") # nfcorpus, fiqa, ...
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_seeds", default=5, type=int)
parser.add_argument("--model_save_path", default="output_retriever", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

# TODO add testing
# TODO add seeds


#### Just some code to print debug information to stdout
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = args.dataset

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
#### Please Note not all datasets contain a dev split, comment out the line if such the case
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

#### Provide any sentence-transformers or HF model
model_name = args.model_name

class ChooseHiddenStateTransformer(models.Transformer):

    def __init__(self, state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
        output_states = self.auto_model(**trans_features, output_hidden_states=True)
        # model truncation
        output_tokens = output_states.hidden_states[self.state]
        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})
        return features

word_embedding_model = ChooseHiddenStateTransformer(args.starting_state, model_name)

if args.pooling_fn == "mean":
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
else:
    raise ValueError("Pooling functions other than mean are not supported")

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
retriever = TrainRetriever(model=model, batch_size=args.train_batch_size)

#### Prepare training samples
train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

#### Training SBERT with cosine-product
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

#### Prepare dev evaluator
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

#### Provide model save path
model_save_path = os.path.join(
    args.model_save_path, 
    args.model_name.replace('/', '-'),
    args.pooling_fn,
    f"{args.starting_state}_to_{args.starting_state + 1}"
)
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = args.num_epochs
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(
    train_objectives=[(train_dataloader, train_loss)], 
    evaluator=ir_evaluator, 
    epochs=num_epochs,
    output_path=model_save_path,
    warmup_steps=warmup_steps,
    optimizer_params={'lr': args.lr, 'eps': 1e-6, 'correct_bias': False},
    weight_decay=args.weight_decay,
    save_best_model=True,
    max_grad_norm=args.max_grad_norm,
)