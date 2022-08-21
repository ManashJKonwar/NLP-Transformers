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
import numpy as np

class TextClassifierDataset:
    def __init__(self, texts, labels):
        # texts: [["hi , my name is Manash"], ["This is a dataset loader for pytorch text classifier"], ...]
        # labels: [["2", "3", ...]]
        self._texts = texts
        self._labels = labels

    def classes(self):
        return self._labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, item):
        # Extracts a specific text and related label for that item 
        text = self._texts[item]
        label = self._labels[item]

        # Tokenizing the sentence
        tok_ip = config.TOKENIZER(
            text,
            add_special_tokens=False,
            padding='max_length',
            max_length=config.MAX_LEN,
            truncation=True,
            return_tensors='pt'
        )

        return tok_ip, np.array(label)