__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import spacy
import torch

from tqdm import tqdm 

import config
from model import TextSummarizerModel

def process_data(article=None, min_sent_length=14):
    # Preprocess article
    article = article.replace("\n","")

    # Load spacy model for segragating sentences
    spacy_model = None
    try:
        spacy_model = spacy.load('en_core_web_lg')
    except OSError:
        spacy.cli.download('en_core_web_lg')
        spacy_model = spacy.load('en_core_web_lg')

    # Generating article sentences
    article_sentences = []
    for sent in spacy_model(article).sents:
        if len(sent) > min_sent_length: 
            article_sentences.append(str(sent))

    return article_sentences

def predict_sentence_score(model, sentence_list, article):
    sentences_tokenized = config.TOKENIZER.batch_encode_plus(
        sentence_list,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        padding="max_length",
        return_token_type_ids=True,
        truncation=True
    )
    sentence_id, sentence_mask = torch.tensor(sentences_tokenized['input_ids'], dtype=torch.long), \
                            torch.tensor(sentences_tokenized['attention_mask'], dtype=torch.long)

    article_tokenized = config.TOKENIZER.batch_encode_plus(
        [article],
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        padding="max_length",
        return_token_type_ids=True,
        truncation=True
    )
    article_id, article_mask = article_tokenized['input_ids'] * len(sentence_list), \
                            article_tokenized['attention_mask'] * len(sentence_list)
    article_id, article_mask = torch.tensor(article_id, dtype=torch.long), \
                        torch.tensor(article_mask, dtype=torch.long)

    preds = model(sentence_id, article_id, sentence_mask, article_mask)
    return preds

if __name__ == "__main__":

    article = """
    Birds are a group of warm-blooded vertebrates constituting the class Aves /ˈeɪviːz/, characterised by feathers, toothless beaked jaws, the laying of hard-shelled eggs, a high metabolic rate, a four-chambered heart, and a strong yet lightweight skeleton. Birds live worldwide and range in size from the 5.5 cm (2.2 in) bee hummingbird to the 2.8 m (9 ft 2 in) ostrich. There are about ten thousand living species, more than half of which are passerine, or “perching” birds. Birds have wings whose development varies according to species; the only known groups without wings are the extinct moa and elephant birds. Wings, which evolved from forelimbs, gave birds the ability to fly, although further evolution has led to the loss of flight in some birds, including ratites, penguins, and diverse endemic island species. The digestive and respiratory systems of birds are also uniquely adapted for flight. Some bird species of aquatic environments, particularly seabirds and some waterbirds, have further evolved for swimming. Birds are feathered theropod dinosaurs and constitute the only known living dinosaurs. Likewise, birds are considered reptiles in the modern cladistic sense of the term, and their closest living relatives are the crocodilians. Birds are descendants of the primitive avialans (whose members include Archaeopteryx) which first appeared about 160 million years ago (mya) in China. According to DNA evidence, modern birds (Neornithes) evolved in the Middle to Late Cretaceous, and diversified dramatically around the time of the Cretaceous–Paleogene extinction event 66 mya, which killed off the pterosaurs and all known non-avian dinosaurs. Many social species pass on knowledge across generations, which is considered a form of culture. Birds are social, communicating with visual signals, calls, and songs, and participating in such behaviours as cooperative breeding and hunting, flocking, and mobbing of predators. The vast majority of bird species are socially (but not necessarily sexually) monogamous, usually for one breeding season at a time, sometimes for years, but rarely for life. Other species have breeding systems that are polygynous (one male with many females) or, rarely, polyandrous (one female with many males). Birds produce offspring by laying eggs which are fertilised through sexual reproduction. They are usually laid in a nest and incubated by the parents. Most birds have an extended period of parental care after hatching. Many species of birds are economically important as food for human consumption and raw material in manufacturing, with domesticated and undomesticated birds being important sources of eggs, meat, and feathers. Songbirds, parrots, and other species are popular as pets. Guano (bird excrement) is harvested for use as a fertiliser. Birds figure throughout human culture. About 120 to 130 species have become extinct due to human activity since the 17th century, and hundreds more before then. Human activity threatens about 1,200 bird species with extinction, though efforts are underway to protect them. Recreational birdwatching is an important part of the ecotourism industry.
    """
    print(article)

    article_sentences = process_data(article=article, min_sent_length=14)
    inferencing_batch_size = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = TextSummarizerModel()
    if use_cuda:
        model.load_state_dict(torch.load(config.MODEL_PATH))
    else:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    model.to(device)

    scores = [] 
    # run predictions using some batch size
    for i in tqdm(range(int(len(article_sentences) / inferencing_batch_size) + 1)):
        batch_start = i* inferencing_batch_size 
        batch_end = (i+1) * inferencing_batch_size if (i+1) * inferencing_batch_size < len(article) else len(article)-1
        batch = article_sentences[batch_start: batch_end]
        if batch:
            preds = predict_sentence_score(model=model, sentence_list=batch, article=article) 
            scores = scores + preds.tolist() 
    
    sent_pred_list = [{"sentence": article_sentences[i], "score": scores[i][0], "index":i} for i in range(len(article_sentences))]
    sorted_sentences = sorted(sent_pred_list, key=lambda k: k['score'], reverse=True) 

    # Selects the top 3 sentences from the entire article
    sorted_result = sorted_sentences[:3] 
    sorted_result = sorted(sorted_result, key=lambda k: k['index']) 
    
    summary = [ x["sentence"] for x in sorted_result]
    summary = " ".join(summary)
    print(summary)