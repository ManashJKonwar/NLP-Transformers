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

def loss_fn(output, target, mask, num_labels):
    """
    This function is utilized to calculate the overall cross entropy loss for predicted
    logits and trace them back to labelled NER label

    args:
    - output: output tensor for text sample from model (may be either for NER tag or POS) which stores the information for each token
    - target: target tensor for text sample from pretrained tokenizer which stores the information for each token
    - mask: attention mask object for text sample
    - num_labels: store the mapping of encoded NER tags / POS tags
    """
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        """
        The main objective of this class is to initialize the pretrained BERT model from either the offline
        bert files or load them from transformer module, calculate the targeted pos tags and ner tags for each 
        token of the text content and finally return the loss for each pass of data points within each epoch
        """
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH, return_dict=False)
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)

    def forward(self, ids, mask, token_type_ids, target_pos, target_tag):
        """
        This method helps to train bert for each textual content, apply dropouts for mitigating the vanishing 
        gradient issue and perform this for ner as well as pos tags and finally returns the cross entropy loss
        for predicted ner and pos tags by the BERT model and validate the same using the labelled information

        args:
        - ids: token ids from BERT pretrained model
        - mask: attention mask from BERT pretrained model for each token
        - token_type_ids: binary mask for identifying textual sentences within a text content
        - target_pos: pos tags for each token
        - target_tag: ner tags for each token

        return:
        - tag: predicted tag for each token
        - pos: predicted pos for each token
        - loss: average cross entropy loss considering detecting ner and pos tags possibly 
        """
        ol, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        bo_tag = self.bert_drop_1(ol)
        bo_pos = self.bert_drop_2(ol)

        tag = self.out_tag(bo_tag)
        pos = self.out_pos(bo_pos)

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return tag, pos, loss