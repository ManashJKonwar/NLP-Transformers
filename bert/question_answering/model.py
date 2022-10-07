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
        super(QuestionAnsweringModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH)
        self.out = nn.Linear(768, 2)

    def forward(self, input_ids, mask, token_type_ids):
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