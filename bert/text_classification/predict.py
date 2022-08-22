__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import joblib
import torch

import config
import dataset
from model import TextClassifierModel

if __name__ == "__main__":

    meta_data = joblib.load(r'output\text_classification\meta.bin')
    enc_labels = meta_data["enc_labels"]

    num_labels = len(list(enc_labels.classes_))
    # ['business', 'entertainment', 'politics', 'sport', 'tech']

    sentence = """
    robinson wants dual code success england rugby union captain jason robinson has targeted dual code success over australia on saturday.  robinson  a former rugby league international before switching codes in 2000  leads england against australia at twickenham at 1430 gmt. and at 1815 gmt  great britain s rugby league team take on australia in the final of the tri-nations tournament.  beating the aussies in both games would be a massive achievement  especially for league   said robinson. england have the chance to seal their third autumn international victory after successive wins over canada and south africa  as well as gaining revenge for june s 51-15 hammering by the wallabies. meanwhile  great britain could end 34 years of failure against australia with victory at elland road. britain have won individual test matches  but have failed to secure any silverware or win the ashes (with a series victory) since 1970.   they have a great opportunity to land a trophy and it would be a massive boost for rugby league in this country if we won   said robinson.  i know the boys can do it - they ve defeated the aussies once already in the tri-nations.  but robinson was not losing sight of the task facing his england side in their final autumn international.  for us  we ve played two and won two this november   he said.  if we beat australia it would be the end to a great autumn series for england. if we stumble then we ll be looking back with a few regrets. robinson also revealed that the union side had sent the great britain team a good luck message ahead of the showdown in leeds.  we signed a card for them today and will write them an email on saturday wishing them all the best   said robinson.  everyone has signed the card - a lot of the guys watch league and we support them fully.  both games will be very tough and hopefully we ll both do well.
    """
    tokenized_sentence = config.TOKENIZER(
        sentence,
        add_special_tokens=False,
        padding='max_length',
        max_length=config.MAX_LEN,
        truncation=True,
        return_tensors='pt'
    )

    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)

    test_dataset = dataset.TextClassifierDataset(
        texts=[sentence], 
        labels=[[3] * len(sentence)]
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = TextClassifierModel(num_labels)
    if use_cuda:
        model.load_state_dict(torch.load(config.MODEL_PATH))
    else:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        input_id = data[0].get('input_ids').squeeze(1).to(device)
        mask = data[0].get('attention_mask').to(device)

        output = model(input_id, mask)

        print(output)