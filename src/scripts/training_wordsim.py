from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
import random
import torch
import numpy as np
import pandas as pd
import argparse
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert-base-cased", type=str)
parser.add_argument("--train_batch_size", default=128, type=int)
parser.add_argument("--starting_state", default=12, type=int)
parser.add_argument("--num_epochs", default=50, type=int)
parser.add_argument("--num_seeds", default=1, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

def fill_examples(df):
    samples = []
    for _, row in df.iterrows():
        ex = InputExample(texts=[row["word1"], row["word2"]], label=row["similarity"])
        samples.append(ex)
    return samples

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

df_train = pd.read_csv("./datasets/word_similarity_dataset_train.csv")
df_val = pd.read_csv("./datasets/word_similarity_dataset_val.csv")
df_test = pd.read_csv("./datasets/word_similarity_dataset_test.csv")

dataset_name = "word_similarity"
model_save_path = "output/" + args.model_name.replace("/", "-") + "_" + dataset_name

logging.info("##### Seed {} #####".format(args.seed))
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_samples = fill_examples(df_train)
dev_samples = fill_examples(df_val)
test_samples = fill_examples(df_test)

word_embedding_model = models.Transformer(args.model_name)

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

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
        output_tokens = output_states.hidden_states[self.state]
        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})
        return features

word_embedding_model = ChooseHiddenStateTransformer(args.starting_state, args.model_name)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=args.device)

curr_model_save_path = model_save_path + f'/seed_{args.seed}' + f'/{args.starting_state}'

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name=f'{dataset_name}-dev')

warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=args.num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=curr_model_save_path
)

del model
word_embedding_model = ChooseHiddenStateTransformer(args.starting_state, args.model_name)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)
word_embedding_model.auto_model.load_state_dict(load_file(curr_model_save_path + "/model.safetensors"), strict=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=args.device)

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name=f'{dataset_name}-test')
test_evaluator(model, output_path=curr_model_save_path)