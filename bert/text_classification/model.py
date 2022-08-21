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

class TextClassifierModel(nn.Module):
    def __init__(self, num_labels, dropout=0.5):
        super(TextClassifierModel, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH, return_dict=False)
        self.linear = nn.Linear(768, self.num_labels)
        self.relu = nn.ReLU()

    def forward(self, ids, mask):
        _, pooled_op = self.bert(ids, attention_mask=mask)

        dropout_op = self.dropout(pooled_op)
        linear_op = self.linear(dropout_op)
        final_layer = self.relu(linear_op)

        return final_layer