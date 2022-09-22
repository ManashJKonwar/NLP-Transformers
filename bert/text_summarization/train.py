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
import pandas as pd

from tqdm import tqdm
from sklearn import model_selection

import config
import dataset

def process_data(articles_path, summaries_path, category_list=['business', 'entertainment', 'politics', 'sport', 'tech']):
    """
    Read individual txt files and structure the contents into a single dataframe
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
    Construct sentence and article dictionary
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
        
if __name__ == "__main__":
    df_summarizer, articles, summaries, categories = process_data(config.ARTICLE_PATH, config.SUMMARY_PATH)

    df_train, df_valid = model_selection.train_test_split(
        df_summarizer, test_size=0.1, random_state=42, stratify=df_summarizer.categories.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_article_dict, train_article_list, test_sentence_dict, test_sentence_list = extract_sentences(
        train_df=df_train, valid_df=df_valid
    )