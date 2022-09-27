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
import dataset
import engine
from model import SentimentAnalyzerModel

if __name__ == "__main__":

    sentence = """
    How this film could be classified as Drama, I have no idea. If I were John Voight and Mary Steenburgen, I would be trying to erase this from my CV. It was as historically accurate as Xena and Hercules. Abraham and Moses got melded into Noah. Lot, Abraham's nephew, Lot, turns up thousands of years before he would have been born. Canaanites wandered the earth...really? What were the scriptwriters thinking? Was it just ignorance (""I remember something about Noah and animals, and Lot and Canaanites and all that stuff from Sunday School"") or were they trying to offend the maximum number of people on the planet as possible- from Christians, Jews and Muslims, to historians, archaeologists, geologists, psychologists, linguists ...as a matter of fact, did anyone not get offended? Anyone who had even a modicum of taste would have winced at this one!
    """

    review = str(sentence)
    review = " ".join(review.split())

    test_dataset = dataset.SentimentAnalyzerDataset(
        reviews=[review], 
        targets=[0]
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=1
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SentimentAnalyzerModel()
    if use_cuda:
        model.load_state_dict(torch.load(config.MODEL_PATH))
    else:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    model.to(device)

    with torch.no_grad():
        for test_ip in test_data_loader:
            mask = test_ip['mask'].to(device)
            input_id = test_ip['ids'].to(device)
            token_type_ids = test_ip['token_type_ids'].to(device)

            output = model(input_id, mask, token_type_ids)
            pos_prediction = torch.sigmoid(output).cpu().detach().numpy()[0][0]
            if pos_prediction<0.5:
                print('Negative')
            else:
                print('Positive')