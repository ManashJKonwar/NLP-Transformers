__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel

if __name__ == "__main__":

    meta_data = joblib.load(r"output\ner\meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    sentence = """
    manash is training custom NER in bangalore
    """
    tokenized_sentence = config.TOKENIZER.encode(sentence)

    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)

    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        pos=[[0] * len(sentence)], 
        tags=[[0] * len(sentence)]
    )

    device = torch.device("cpu")
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _ = model(**data)

        print(
            enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
        print(
            enc_pos.inverse_transform(
                pos.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )