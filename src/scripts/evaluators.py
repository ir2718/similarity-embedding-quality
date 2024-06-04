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

    def test_mode(self):
        pass

class SentencePairEvaluator:

    def __init__(self):
        self.best_mean_ap = None
        self.best_f1 = None
        self.best_f1_threshold = None
        self.best_acc = None
        self.best_acc_threshold = None
        self.test = False

    def __call__(self, embeddings, labels):
        embeddings_np = embeddings.sigmoid().detach().cpu().numpy()
        labels = labels.astype(np.int32)

        mean_ap = average_precision_score(labels, embeddings_np)

        if not self.test:
            acc, best_acc_threshold = SentencePairEvaluator.find_best_acc_and_threshold(embeddings_np, labels, True)
            f1, precision, recall, best_f1_threshold = SentencePairEvaluator.find_best_f1_and_threshold(embeddings_np, labels, True)

            if self.best_mean_ap is None or mean_ap > self.best_mean_ap:
                self.best_mean_ap = mean_ap
            
                self.best_f1 = f1
                self.best_f1_threshold = best_f1_threshold

                self.best_acc = acc
                self.best_acc_threshold = best_acc_threshold

        else:
            # in test mode the best threshold are used
            acc = accuracy_score(labels, embeddings_np > self.best_acc_threshold)
            f1 = f1_score(labels, embeddings_np > self.best_f1_threshold)
            recall = recall_score(labels, embeddings_np > self.best_f1_threshold, zero_division=0.0)
            precision = precision_score(labels, embeddings_np > self.best_f1_threshold, zero_division=0.0)

        return {
            "map": mean_ap,
            "acc": acc,
            "f1": f1,
            "recall": recall,
            "precision": precision
        }

    def test_mode(self):
        self.test = True

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
    

class NLIEvaluator:

    def __init__(self):
        self.best_f1 = None
        self.best_f1_threshold = None
        self.best_acc = None
        self.best_acc_threshold = None
        self.test = False

    def __call__(self, embeddings, labels):
        labels = labels.astype(np.int32)

        if len(embeddings.shape) == 2 and embeddings.shape[1] == 3:
            embeddings_np = embeddings.argmax(axis=1).detach().cpu().numpy()

            acc = accuracy_score(labels, embeddings_np)
            f1 = f1_score(labels, embeddings_np, average="macro")
            recall = recall_score(labels, embeddings_np, average="macro", zero_division=0.0)
            precision = precision_score(labels, embeddings_np, average="macro", zero_division=0.0)
        
        else:
            embeddings_np = embeddings.sigmoid().detach().cpu().numpy()

            # use thresholding in case of binary
            if not self.test:
                acc, best_acc_threshold = SentencePairEvaluator.find_best_acc_and_threshold(embeddings_np, labels, True)
                f1, precision, recall, best_f1_threshold = SentencePairEvaluator.find_best_f1_and_threshold(embeddings_np, labels, True)

                if self.best_f1 is None or f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_f1_threshold = best_f1_threshold

                    self.best_acc = acc
                    self.best_acc_threshold = best_acc_threshold

            else:
                # in test mode the best threshold are used
                acc = accuracy_score(labels, embeddings_np > self.best_acc_threshold)
                f1 = f1_score(labels, embeddings_np > self.best_f1_threshold)
                recall = recall_score(labels, embeddings_np > self.best_f1_threshold, zero_division=0.0)
                precision = precision_score(labels, embeddings_np > self.best_f1_threshold, zero_division=0.0)

        return {
            "acc": acc,
            "f1": f1,
            "recall": recall,
            "precision": precision
        }

    def test_mode(self):
        self.test = True