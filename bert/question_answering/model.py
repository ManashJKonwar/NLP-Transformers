__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import config
import transformers
import torch.nn as nn

class QuestionAnsweringModel(nn.Module):
    def __init__(self):
        """
        The main objective of this class is to initialize the pretrained BERT model from either the offline
        bert files or load them from transformer module, calculate start and end logits for the answer
        predicted for context and question combination from the sequence output layer of BERT and finally
        return these values
        """
        super(QuestionAnsweringModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH, return_dict=False)
        self.out = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        """
        This method helps to train bert for each textual content (context + question) and finally returns the predicted
        start and end logits for probable answer word or phrase within the context by the BERT model.

        args:
        - ids: token ids from BERT pretrained model
        - mask: attention mask from BERT pretrained model for each token
        - token_type_ids: binary mask for identifying textual sentences within a text content

        return:
        - start_logits: predicted start position for predicted answer word or phrase from sequenced input
        - end_logits: predicted end position for predicted answer word or phrase from sequenced input
        """
        # o1: sequence_output (batch_size, num_tokens, 768) --> (batch_size, num_tokens, 2)
        # o2: pooled_output
        o1, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # (batch_size, num_tokens, 768)
        logits = self.out(o1)
        # (batch_size, num_tokens, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # (batch_size, num_tokens), (batch_size, num_tokens)
        
        return start_logits, end_logits