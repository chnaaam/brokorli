def calculate_em_score(true_text_list, pred_text_list):
    """
    Calculate the exact matching score for the machine reading comprehension task
    """
    # return true_text == pred_text
    
    em_count = 0
    for true_text, pred_text in zip(true_text_list, pred_text_list):
        if true_text == pred_text:
            em_count += 1
            
    return em_count / len(true_text_list)