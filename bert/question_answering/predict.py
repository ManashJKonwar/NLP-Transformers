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
from model import QuestionAnsweringModel

if __name__ == "__main__":

    context = """
    My home is in bengaluru and i am currently testing question answering module.
    """
    question = """
    Which module am i currently working with?
    """

    inputs = config.TOKENIZER.encode_plus(
        question,
        context, 
        max_length=config.MAX_LEN,
        padding="max_length",
        truncation="only_second",
        return_tensors="pt",
        return_offsets_mapping=True,
        return_token_type_ids=True  
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = QuestionAnsweringModel()
    if use_cuda:
        model.load_state_dict(torch.load(config.MODEL_PATH))
    else:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    model.to(device)

    with torch.no_grad():
        start_logits, end_logits = model(
            ids=inputs['input_ids'],
            mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )

        all_tokens = config.TOKENIZER.convert_ids_to_tokens(inputs['input_ids'][0].tolist())

        start_index = 0
        answer_tokens = all_tokens[torch.argmax(start_logits[0][start_index:]) : torch.argmax(end_logits[0][start_index:])+1]
        answer = config.TOKENIZER.decode(config.TOKENIZER.convert_tokens_to_ids(answer_tokens))

        print('Answer: %s' %(str(answer)))