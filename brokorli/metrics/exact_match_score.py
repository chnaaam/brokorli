def calculate_em_score(true_answers, pred_answers):
    """
    Calculate the exact matching score for the machine reading comprehension task
    """
    
    em_count = 0
    for true_answer, pred_answer in zip(true_answers, pred_answers):
        if true_answer == pred_answer:
            em_count += 1
            
    return em_count / len(true_answers)