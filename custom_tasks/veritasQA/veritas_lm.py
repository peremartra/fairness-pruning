def parse_veritasqa_choices(doc):
    """Parse and clean VeritasQA answer choices.
    
    VeritasQA stores answers as semicolon-separated strings.
    This function splits them, cleans whitespace, and returns
    correct answers first (so index 0 is always correct).
    
    Args:
        doc: Dataset example with 'correct_answers' and 'incorrect_answers'
        
    Returns:
        list: All answer choices (correct + incorrect)
    """
    correct = [doc['best_answer'].strip()] 
    
    incorrect = [ans.strip() for ans in doc['incorrect_answers'].split(';')]
    return correct + incorrect
