from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, recall_score, precision_score
import numpy as np

class STSEvaluator:

    def __call__(self, embeddings, labels):
        embeddings_np = embeddings.detach().cpu().numpy()
        return {
            "spearman": spearmanr(embeddings_np, labels)[0],
            "pearson": pearsonr(embeddings_np, labels)[0]
        }

class SentencePairEvaluator:

    def __call__(self, embeddings, labels):
        embeddings_np = embeddings.sigmoid().detach().cpu().numpy()
        labels = labels.astype(np.int32)

        mean_ap = average_precision_score(labels, embeddings_np)

        acc, _ = SentencePairEvaluator.find_best_acc_and_threshold(embeddings_np, labels, True)
        f1, precision, recall, _ = SentencePairEvaluator.find_best_f1_and_threshold(embeddings_np, labels, True)

        return {
            "map": mean_ap,
            "acc": acc,
            "f1": f1,
            "recall": recall,
            "precision": precision
        }

    # taken from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/evaluation/evaluators/PairClassificationEvaluator.py
    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(np.array(labels) == 0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold