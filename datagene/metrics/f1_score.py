from seqeval.metrics import f1_score
from seqeval.scheme import IOBES

def calculate_f1_score(true_y, pred_y):
    """
    BIOES 태깅된 Sequence Labeling Task에 대한 F1 Score를 계산하는 함수입니다.
    """
    return f1_score(true_y, pred_y, mode="strict", scheme=IOBES)