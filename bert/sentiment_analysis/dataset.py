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

class SentimentAnalyzerDataset:
    def __init__(self, reviews, targets):
        # reviews: [["hi , my name is Manash and the day is really hectic"], ["This is a dataset loader for pytorch which is awesome for training sentiment"], ...]
        # targets: [[0], [1], ...]
        self._reviews = reviews
        self._targets = targets

    def __len__(self):
        return len(self._reviews)

    def __getitem__(self, item):
        # Extracts specific reviews and related targets
        review = str(self._reviews[item])
        review = " ".join(review.split())

        inputs = config.TOKENIZER.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            truncation=True
        )
        
        # Extracting BERT ids, attention mask and token ids
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # Padding input
        padding_len = config.MAX_LEN - len(ids)

        # Adding padding to make the list == MAX LENGTH from config
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self._targets[item], dtype=torch.float),
        }