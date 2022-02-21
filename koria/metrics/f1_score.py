from seqeval.metrics import f1_score
from seqeval.scheme import IOBES

def calculate_f1_score(true_y, pred_y):
    """
    Calculate the F1 score for the BIOES tagged sequence labeling task.
    """
    return f1_score(true_y, pred_y, mode="strict", scheme=IOBES)