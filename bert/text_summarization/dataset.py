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
        """
        The main objective of this class is to structure each text data point for extractive text summarization operation
        and to convert the collection of article and subsequent summary data into tokenized forms using the pretrained tokenizer 
        extracted BERT attributes which will be equal to MAX_LEN for training process as well as validation
        """
        # data: pandas dataframe consisting of important columns ("docs" for relevant article and "sents" for summarized context)
        self._data = data

    def __len__(self):
        """
        This is a getter method responsible to extract the number of textual datapoints within each 
        training sample
        """
        return len(self._data)

    def __getitem__(self, item):
        """
        This is a getter method responsible for extracting textual matrices which will be utilize to fine
        tune the base BERT model for generating extractive summaries by scoring each sentence within an article.

        args:
        - item (int): index of item to be considered for tokenization within the training sample data 

        return: 
        - (dict)" collection of tensors for the selected textual content (summary + article)
        """
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