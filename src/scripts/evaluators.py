from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, recall_score, precision_score
import numpy as np

class STSEvaluator:

    def __call__(self, embeddings, scores):
        embeddings_np = embeddings.detach().cpu().numpy()
        return {
            "spearman": spearmanr(embeddings_np, scores)[0],
            "pearson": pearsonr(embeddings_np, scores)[0]
        }

class SentencePairEvaluator:

    def __call__(self, embeddings, scores):
        embeddings_np = embeddings.sigmoid().detach().cpu().numpy()
        scores = scores.astype(np.int32)

        preds = (embeddings_np > 0.5).astype(np.int32)

        mean_ap = average_precision_score(scores, embeddings_np)
        acc = accuracy_score(scores, preds)
        f1 = f1_score(scores, preds)
        recall = recall_score(scores, preds, zero_division=0.0)
        precision = precision_score(scores, preds, zero_division=0.0)

        return {
            "map": mean_ap,
            "acc": acc,
            "f1": f1,
            "recall": recall,
            "precision": precision
        }

