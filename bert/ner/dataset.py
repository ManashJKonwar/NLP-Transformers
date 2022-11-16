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

class EntityDataset:
    def __init__(self, texts, pos, tags):
        """
        The main objective of this class is to structure each text data point for NER operation
        and to convert the collection of text data and using the pretrained tokenizer extract
        BERT attributes which will be equal to MAX_LEN for training process as well as validation
        """
        # texts: [["hi , my name is Manash"], ["This is a dataset loader for pytorch"], ...]
        # pos/tags: [[1 2 3 4 1 5], [....], ...]
        self._texts = texts
        self._pos = pos
        self._tags = tags

    def __len__(self):
        """
        This is a getter method responsible to extract the number of textual datapoints within each 
        training sample
        """
        return len(self._texts)

    def __getitem__(self, item):
        """
        This is a getter method responsible for extracting textual matrices which will be utilize to fine
        tune the base BERT model for predicting NERs

        args:
        - item (int): index of item to be considered for tokenization within the training sample data 

        return: 
        - (dict)" collection of tensors for the selected textual content
        """
        # Extracts a specific text and related pos/tags for that item 
        text = self._texts[item]
        pos = self._pos[item]
        tags = self._tags[item]

        # Tokenizing the sentences
        ids = []
        target_pos = []
        target_tag = []

        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            # If text is not in vocabulary of tokenizer
            # Manash: ma #na #sh
            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tags[i]] * input_len)

        # Padding the text
        ids = ids[:config.MAX_LEN - 2] # -2 is done for special tokens
        target_pos = target_pos[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        ids = [101] + ids + [102] # 101 and 102 represents [CLS] and [SEP] token
        target_pos = [0] + target_pos + [0] # adding 0 to replicate CLS and SEP addition in pos
        target_tag = [0] + target_tag + [0] # adding 0 to replicate CLS and SEP addition in tags

        # Attention mask
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        # Padding input
        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }