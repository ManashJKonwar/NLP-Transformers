__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import torch
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    This function is utilize to train the BERT architecture for the training sample and 
    also calculate the loss for this epoch, store the relevant loss and at the end return 
    the averaged out loss for the train sample

    args: 
    - data_loader: pytorch data loader object for the train sample
    - model: BERT model which will be trained for detecting NERs
    - optimizer: Adam optimizer to adjust after each data pass
    - device: pytorch device object to consider for training
    - scheduler: schedule object with a learning rate that decreases linearly from the initial lr set in the optimizer to 0

    return:
    - averaged out cross entropy loss for the entire train sample
    """
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)

def eval_fn(data_loader, model, device):
    """
    This function is utilize to evaluate the BERT architecture for the validation sample and 
    also calculate the loss for this epoch, store the relevant loss and at the end return 
    the averaged out loss for the validation sample

    args: 
    - data_loader: pytorch data loader object for the validation sample
    - model: trained BERT model which will be validated for detecting NERs
    - device: pytorch device object to consider for validation

    return:
    - averaged out cross entropy loss for the entire validation sample
    """
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        _, _, loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)