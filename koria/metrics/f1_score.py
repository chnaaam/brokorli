import collections

from seqeval.metrics import f1_score
from seqeval.scheme import IOBES

# F1 score for sequence labeling
def calculate_sl_f1_score(true_y, pred_y):
    """
    Calculate the F1 score for the BIOES tagged sequence labeling task.
    """
    return f1_score(true_y, pred_y, mode="strict", scheme=IOBES)

# F1 score for question answering
def calculate_qa_f1_score(true_answers, pred_answers):
    
    f1_score_list = []
    
    for true_answer, pred_answer in zip(true_answers, pred_answers):
        
        # refer to huggingface compute_f1 function
        common_tokens = collections.Counter(list(true_answer)) & collections.Counter(list(pred_answer))
        num_same = sum(common_tokens.values())
        
        if num_same == 0 or true_answer == "" or pred_answer == "":
            f1_score_list.append(0)
            continue

        precision = 1.0 * num_same / len(pred_answer)
        recall = 1.0 * num_same / len(true_answer)
        f1_score = (2 * precision * recall) / (precision + recall)
        
        f1_score_list.append(f1_score)
    
    return sum(f1_score_list) / len(f1_score_list)
    