from seqeval.metrics import f1_score
from seqeval.scheme import IOBES

def calculate_f1_score(true_y, pred_y):
    return f1_score(true_y, pred_y, mode="strict", scheme=IOBES)