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

    """

    review = str(sentence)
    review = " ".join(review.split())

    inputs = config.TOKENIZER.encode_plus(
        review, 
        None, 
        add_special_tokens=True, 
        max_length=config.MAX_LEN
    )

    test_dataset = dataset.SentimentAnalyzerDataset(
        reviews=[review], 
        targets=[0]
    )

    test_data_loader = torch.utils.data.Dataloader(
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