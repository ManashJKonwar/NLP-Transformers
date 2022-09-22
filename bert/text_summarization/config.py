__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "pretrained_models/bert/bert_base_uncased"
MODEL_PATH = "output/text_summarization/model.bin"
ARTICLE_PATH = "input/text_summarization/news_articles"
SUMMARY_PATH = "input/text_summarization/summaries"
TRAINING_FILE = "input/text_summarization/text_summarization_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)