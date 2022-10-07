__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Takes 2 lists of strings and compares corresponding elements to check if they are exact matches. 
def exact_match(preds,answer):
    exact_matches = []
    for i in range(len(preds)) : 
        exact_matches.append(normalize_text(preds[i]) == normalize_text(answer[i]))
    return exact_matches

# Takes 2 lists of strings and calculates the F1 scores between corresponding elements in both strings
def f1_score(preds,answer):
    f1_scores = []
    for i in range(len(preds)) : 
        shared_words = 0
        pred_words = normalize_text(preds[i]).split()
        answer_words = normalize_text(answer[i]).split()
        shared_words = set(pred_words) & set(answer_words)
        try : 
            precision = (len(shared_words)/len(pred_words))
        except : 
            precision = 0
        try : 
            recall = (len(shared_words)/len(answer_words))
        except :
            recall = 0
        
        if(precision == 0 or recall == 0) : 
            f1_scores.append(0)
        else : 
            f1_scores.append(2 * (precision * recall)/ (precision + recall))
    return f1_scores

# Getting the predicted answers for a given batch
def get_batch_predictions(example_batch,batch_start_probs,batch_end_probs,batch_size = 8):
    pred_answers = []
    context_start_indices = example_batch["Context_start_index"]
    context_end_indices = example_batch["Context_end_index"]
    for i in range(batch_size) :
        instance_start_probs,instance_end_probs = batch_start_probs[i],batch_end_probs[i]
        context_start,context_end = context_start_indices[i],context_end_indices[i]
        offset_maps = example_batch["Offset_mapping"][i]
        context = example_batch["Context"][i]
        best_start,best_end,best_prob = context_start,context_start,instance_start_probs[context_start] * instance_end_probs[context_start]
        for j in range(context_start,context_end + 1) : 
            for k in range(j,context_end + 1) : 
                current_prob = instance_start_probs[j] * instance_end_probs[k]
                if(current_prob > best_prob) : 
                    best_start,best_end,best_prob = j,k,current_prob
        start_char = offset_maps[best_start][0]
        end_char = offset_maps[best_end][1]
        ans = context[start_char:end_char]
        pred_answers.append(ans)

    return pred_answers