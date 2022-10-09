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

class QuestionAnsweringDataset:
    def __init__(self, context, question, answers):
        # context: [["This is the first context..."], ["This is the second context..."], ....]
        # question: [["what is the main context for the first context?"], ["What is the main context for the second context"], ....]
        # answers: [[{"text": "Answer1_text", "answer_start": 500, "answer_end": 510}], [{"text": "Answer1_text", "answer_start": 600, "answer_end": 615}]]
        self._context = context
        self._question = question
        self._answers = answers

    def __len__(self):
        return len(self._context)

    def __getitem__(self, item):
        # Extracts specific questions and also relevant context
        context = str(self._context[item])
        context = " ".join(context.split())

        question = str(self._question[item])
        question = " ".join(question.split())

        answers = self._answers[item]

        # inputs = config.TOKENIZER.encode(
        #     [context, question],
        #     # max_length=config.MAX_LEN,
        #     # padding="max_length",
        #     # truncation=True,
        #     # return_tensors="pt",
        # )
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

        # Extracting BERT ids, attention mask, offset mapping and token_type_ids
        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        offset_mapping = inputs["offset_mapping"]
        token_type_ids = inputs["token_type_ids"]

        # Initialize lists to contain the token indices of answer start/end
        context_start_idx = 0
        context_end_idx = len(offset_mapping[0]) - 1
        sequence_ids = inputs.sequence_ids()
        cls_index = input_ids[0][config.TOKENIZER.cls_token_id]
        
        while(sequence_ids[context_start_idx] != 1) :
            context_start_idx += 1
        while(sequence_ids[context_end_idx] != 1) :
            context_end_idx -= 1
        
        if not (offset_mapping[0][context_start_idx][0] <= answers[0]['answer_start'] and offset_mapping[0][context_end_idx][1] >= answers[0]['answer_end']) :
            inputs["start_positions"] = (cls_index)
            inputs["end_positions"] = (cls_index)
        else :
            current_token = context_start_idx
            gotStart, gotEnd = False,False

            for start_char,end_char in (offset_mapping[0][context_start_idx : context_end_idx  + 1]) :  
                if (start_char == answers[0]['answer_start']) :
                    inputs["start_positions"] = current_token
                    gotStart = True
                if (end_char == answers[0]['answer_end']) : 
                    inputs["end_positions"] = current_token
                    gotEnd = True
                current_token += 1

            if (gotStart == False) :
                inputs["start_positions"] = (cls_index)
            if (gotEnd == False) :
                inputs["end_positions"] = (cls_index)

        # Form binary target start and end position tensors
        targets = [0] * config.MAX_LEN
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        if inputs["start_positions"] is not None and inputs["end_positions"] is not None:
            targets_start[inputs["start_positions"]] = 1
            targets_end[inputs["end_positions"]] = 1

        return {
            "context": context,
            "answers": answers[0]['text'],
            "input_ids": input_ids,
            "mask": mask,
            "offset_mapping": offset_mapping,
            "token_type_ids": token_type_ids,
            "context_start_idx": context_start_idx,
            "context_end_idx": context_end_idx,
            "start_positions": inputs["start_positions"],
            "end_positions": inputs["end_positions"],
            "targets_start": torch.tensor(targets_start, dtype = torch.long),
            "targets_end": torch.tensor(targets_end, dtype = torch.long)
        }