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
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 10
BASE_MODEL_PATH = "pretrained_models/bert/bert_base_uncased"
MODEL_PATH = "output/question_answering/model.bin"
QnA_TRAINING_PATH = "input/question_answering/train-v1.1.json"
QnA_VALIDATION_PATH = "input/question_answering/dev-v1.1.json"
TRAINING_FILE = "input/question_answering/question_answering_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)