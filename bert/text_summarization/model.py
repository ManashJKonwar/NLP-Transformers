__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import config
import torch
import transformers
import torch.nn as nn

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class TextSummarizerModel(nn.Module):
    def __init__(self, n_feature=768, dropout=0.3):
        """
        The main objective of this class is to initialize the pretrained BERT model from either the offline
        bert files or load them from transformer module, calculate the article and summary based embeddings,
        concatenate them and finally apply, reduce the embeddings to 768D, apply relu activation, apply dropout
        for reducing vanishing gradients and finally calculating the probability of each sentence for ranking
        them into top k attributes for summarization.
        """
        super(TextSummarizerModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH) 
        self.pre_classifier = nn.Linear(n_feature*3, 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)
        self.classifierSigmoid = nn.Sigmoid()

    def forward(self, sent_ids, doc_ids, sent_mask, doc_mask):
        """
        This method helps to train bert for each textual content (summary + article) and finally returns the predicted scores
        of each sentence being in top k ranks of important sentences catching the relevance of the main article.

        args:
        - sent_ids: token ids for summarized sentences from BERT pretrained model
        - doc_ids: token ids for article from BERT pretrained model
        - sent_mask: attention mask from BERT pretrained model for each token within the summarized context
        - doc_mask: ate

        return:
        - final layer: tensor representing sigmoid scores of each sentence withing the context based on provided summary
        """
        sent_output = self.bert(input_ids=sent_ids, attention_mask=sent_mask) 
        sentence_embeddings = mean_pooling(sent_output, sent_mask) 
        
        article_output = self.bert(input_ids=doc_ids, attention_mask=doc_mask) 
        article_embeddings = mean_pooling(article_output, doc_mask)

        # Elementwise product of sentence embeddings and article embeddings
        combined_features = sentence_embeddings * article_embeddings  

        # Concatenate input features and their elementwise product
        concat_features = torch.cat((sentence_embeddings, article_embeddings, combined_features), dim=1)   
        
        pooler = self.pre_classifier(concat_features) 
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        final_layer = self.classifierSigmoid(output) 

        return final_layer