__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
import spacy
import pickle
import numpy as np
import pandas as pd

import torch

from tqdm import tqdm
from sklearn import model_selection
from rouge_score import rouge_scorer 

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import TextSummarizerModel

def process_data(articles_path, summaries_path, category_list=['business', 'entertainment', 'politics', 'sport', 'tech']):
    """
    This function is responsible for combining individual txt based training samples and structuring them into a single dataframe which
    would be utilized to create the multi class text classifier.  

    args:
    - articles_path (str): base folder path for articles
    - summaries_path (str): base folder path for summaries
    - categories_list (list, str): classes to be considered for classifier training.

    return:
    - df: pandas dataframe consisting of articles, summaries and category columns
    - articles: list of articles
    - summaries: list of summaries for subsequent articles
    - categories: lis of categories for subsequent articles   
    """
    articles, summaries, categories = [], [], []

    if os.path.exists(config.TRAINING_FILE):
        df = pd.read_csv(config.TRAINING_FILE)
        return df, df.articles.tolist(), df.summaries.tolist(), df.categories.tolist()
    else:
        # Read raw files and form the final dataframe with articles and respective summaries
        for category in category_list:
            article_file_paths = [os.path.join(articles_path, category, file) for file in os.listdir(os.path.join(articles_path, category)) if file.endswith(".txt")]
            summary_file_paths = [os.path.join(summaries_path, category, file) for file in os.listdir(os.path.join(summaries_path, category)) if file.endswith(".txt")]

            print(f'found {len(article_file_paths)} file in articles/{category} folder, {len(summary_file_paths)} file in summaries/{category} folder')

            if len(article_file_paths) != len(summary_file_paths):
                print('number of files is not equal') 
                return    

            for idx_file in range(len(article_file_paths)):
                categories.append(category)
                with open(article_file_paths[idx_file], mode = 'r', encoding = "ISO-8859-1") as file:
                    articles.append(file.read())
                with open(summary_file_paths[idx_file], mode = 'r', encoding = "ISO-8859-1") as file:
                    summaries.append(file.read())

            print(f'total {len(articles)} file in articles folders, {len(summaries)} file in summaries folders')
        
        df = pd.DataFrame({'articles':articles,'summaries': summaries, 'categories' : categories})
        df.to_csv(config.TRAINING_FILE, index=False) 
        return df, articles, summaries, categories

def extract_sentences(train_df, valid_df):
    """
    This function is responsible to construct article and summary based dictionaries
    
    args:
    - train_df: training dataframe for which articles and summaries needs to broken into collection of sentences
    - valid_df: validation dataframe for which articles and summaries needs to be broken into collection of sentences

    return:
    - train_article_dict: dictionary of article and summaries with unique indexing for training data
    - train_sentence_list: list of sentences withim each article and summary combination for training data
    - test_article_dict: dictionary of article and summaries with unique indexing for validation data
    - test_sentence_list: list of sentences withim each article and summary combination for validation data
    """
    if os.path.exists(os.path.join('input', 'text_summarization', 'train_article_dict.pkl')) and \
    os.path.exists(os.path.join('input', 'text_summarization', 'train_sentence_list.pkl')) and \
    os.path.exists(os.path.join('input', 'text_summarization', 'test_article_dict.pkl')) and \
    os.path.exists(os.path.join('input', 'text_summarization', 'test_sentence_list.pkl')):
        with open(os.path.join('input', 'text_summarization', 'train_article_dict.pkl'), 'rb') as handle:
            train_article_dict = pickle.load(handle)
        with open(os.path.join('input', 'text_summarization', 'train_sentence_list.pkl'), 'rb') as handle:
            train_sentence_list = pickle.load(handle)
        with open(os.path.join('input', 'text_summarization', 'test_article_dict.pkl'), 'rb') as handle:
            test_article_dict = pickle.load(handle)
        with open(os.path.join('input', 'text_summarization', 'test_sentence_list.pkl'), 'rb') as handle:
            test_sentence_list = pickle.load(handle)
        
        return train_article_dict, train_sentence_list, test_article_dict, test_sentence_list
    else:
        spacy_model = None
        try:
            spacy_model = spacy.load('en_core_web_lg')
        except OSError:
            spacy.cli.download('en_core_web_lg')
            spacy_model = spacy.load('en_core_web_lg')

        # Extracting sentences with spacy
        def get_dicts(df, folder="test"):   
            sents_dict = {}
            doc_dict = { i: {"article": df.articles[i], "summary": df.summaries[i]} for i in df.index }
            raw_docs = [ doc_dict[k]["article"] for k in doc_dict.keys()]

            doc_sents = {}
            sents_list = []
            raw_sents = [] 
            i = 0
            min_sent_length = 14
            for k in tqdm(doc_dict.keys()):
                article = doc_dict[k]["article"]  
                highlight = doc_dict[k]["summary"] 
                sents = spacy_model(article).sents
                doc_sent_ids = [] 
                for sent in sents:
                    if (len(sent)) > min_sent_length:
                        sents_dict[i] = {"docid":k, "text": str(sent)} 
                        sents_list.append({"sentid":i, "docid":k, "text": str(sent) }) 
                        raw_sents.append(str(sent))
                        i += 1  
            return doc_dict, sents_list

        train_article_dict, train_sentence_list = get_dicts(train_df)
        test_article_dict, test_sentence_list = get_dicts(valid_df)

        write_dict={
            'train_article_dict': train_article_dict,
            'train_sentence_list': train_sentence_list,
            'test_article_dict': test_article_dict,
            'test_sentence_list': test_sentence_list
        }
        for write_key in write_dict.keys():
            with open(os.path.join('input', 'text_summarization', write_key+'.pkl'), 'wb') as out_file:
                pickle.dump(write_dict[write_key], out_file, protocol=pickle.HIGHEST_PROTOCOL)

        return train_article_dict, train_sentence_list, test_article_dict, test_sentence_list

def get_final_data(train_article_dict, train_sentence_list, test_article_dict, test_sentence_list):
    """
    Extracts labels and balanced dataset by performing 3 operations
    1. Extract labels for each sentence in sentence list
    2. If dataset is imbalanced, i.e. most sentences are unlikely to be in the respective summary
    3. Construct new dataset of examples that balances positive examples with negative examples
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def get_rougue_score(text, highlights, metric="rougeL"):
        max_score = 0
        for h_text in highlights:
            score =  scorer.score(text, h_text)[metric].fmeasure
            # print(score, text, "\n \t" , h_text)
            if score > max_score:
                max_score = score 
        return max_score 

    def get_label(sent, doc_dict,  score_threshold = 0.55):
        sent_id, doc_id, sentence = sent["sentid"], sent["docid"], sent["text"]  
        summaries = doc_dict[doc_id]["summary"].split("\n")
        doc = doc_dict[doc_id]["article"]

        label_score = get_rougue_score(sentence, summaries) 
        # Normalize label to 0/1 based on rogue score threshold
        label_score = 0 if label_score < score_threshold else 1 
        return (sentence, doc, label_score)
        
    def sub_sample(sents_batch, doc_dict, neg_multiplier=2):
        # get labels 
        vals = [get_label(x, doc_dict)  for x in tqdm(sents_batch, desc='Extracting rouge scores for sentence list')] 

        # construct arrays of sentences, corresponding documents and labels  
        sents, docs, y = [], [], [] 
        for row in vals:
            sents.append(row[0])
            docs.append(row[1])
            y.append(row[2])

        # get balanced number of positive and negative
        sub_df = pd.DataFrame.from_dict({"sents":sents, "docs":docs, "y":y}) 
        pos_df = sub_df[sub_df.y == 1]
        neg_df = sub_df[sub_df.y == 0]

        print("Negative sample size:", len(neg_df))
        print("Positive sample size:", len(pos_df))

        sub_neg_df = neg_df.sample(len(pos_df)*neg_multiplier) 
        balanced_df = pd.concat([pos_df, sub_neg_df]).reset_index(drop=True)
        
        return balanced_df

    if os.path.exists(os.path.join('input', 'text_summarization', 'train_balanced.csv')) and \
    os.path.exists(os.path.join('input', 'text_summarization', 'valid_balanced.csv')):
        train_balanced_df = pd.read_csv(os.path.join('input', 'text_summarization', 'train_balanced.csv'))
        test_balanced_df = pd.read_csv(os.path.join('input', 'text_summarization', 'valid_balanced.csv'))
    else:
        train_balanced_df = sub_sample(train_sentence_list, train_article_dict)
        test_balanced_df = sub_sample(test_sentence_list, test_article_dict)

        train_balanced_df.to_csv(os.path.join('input', 'text_summarization', 'train_balanced.csv'), index=False)
        test_balanced_df.to_csv(os.path.join('input', 'text_summarization', 'valid_balanced.csv'), index=False)

    return train_balanced_df, test_balanced_df
        
if __name__ == "__main__":
    df_summarizer, articles, summaries, categories = process_data(config.ARTICLE_PATH, config.SUMMARY_PATH)

    df_train, df_valid = model_selection.train_test_split(
        df_summarizer, test_size=0.1, random_state=42, stratify=df_summarizer.categories.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_article_dict, train_article_list, valid_sentence_dict, valid_sentence_list = extract_sentences(
        train_df=df_train, valid_df=df_valid
    )

    df_train_balanced, df_valid_balanced = get_final_data(train_article_dict, train_article_list, valid_sentence_dict, valid_sentence_list)

    train_dataset = dataset.TextSummarizerDataset(
        data=df_train_balanced
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4, shuffle=True
    )

    valid_dataset = dataset.TextSummarizerDataset(
        data=df_valid_balanced
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1, shuffle=True
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = TextSummarizerModel()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_acc, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_acc, test_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss