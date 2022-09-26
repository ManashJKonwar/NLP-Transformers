__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import torch
import config

class TextSummarizerDataset:
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        # Extracts specific articles and there summary
        sentence = str(self._data.iloc[item].sents)
        sentence = " ".join(sentence.split())
        
        article = str(self._data.iloc[item].docs)
        article = " ".join(article.split())

        inputs = config.TOKENIZER.batch_encode_plus(
            [sentence, article],
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )

        # Extracting BERT ids, attention mask and token ids
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "sent_ids": torch.tensor(ids[0], dtype=torch.long),
            "doc_ids": torch.tensor(ids[1], dtype=torch.long),
            "sent_mask": torch.tensor(mask[0], dtype=torch.long),
            "doc_mask": torch.tensor(mask[1], dtype=torch.long),
            "targets": torch.tensor([self._data.iloc[item].y], dtype=torch.long)
        } 