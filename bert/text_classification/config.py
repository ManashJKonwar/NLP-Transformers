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
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-6
BASE_MODEL_PATH = "pretrained_models/bert/bert_base_cased"
MODEL_PATH = "output/text_classification/model.bin"
TRAINING_FILE = "input/text_classification/text_classification_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH
)