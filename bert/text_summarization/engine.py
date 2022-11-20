__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import config
import torch
import torch.nn as nn
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    This function is utilize to train the BERT architecture for the training sample and 
    also calculate the loss for this epoch, store the relevant cross entropy loss and at the end return 
    averaged out accuracy and loss for the train sample considering the predicted score for each sentence 
    within the article followed by the actual top k scored sentences.

    args: 
    - data_loader: pytorch data loader object for the train sample
    - model: BERT model which will be trained for predicted start and end logits for answers to relevant questions
    - optimizer: Adam optimizer to adjust after each data pass
    - device: pytorch device object to consider for training
    - scheduler: schedule object with a learning rate that decreases linearly from the initial lr set in the optimizer to 0

    return:
    - total_acc_train / nb_tr_examples: averaged out accuracy for the entire train sample
    - total_loss_train / len(data_loader): averaged out loss for the entire train sample
    """
    model.train()
    total_acc_train = 0
    total_loss_train = 0
    nb_tr_examples = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        sent_ids = data['sent_ids']
        doc_ids = data['doc_ids']
        sent_mask = data['sent_mask']
        doc_mask = data['doc_mask'] 
        targets = data['targets'] 

        sent_ids = sent_ids.to(device, dtype = torch.long)
        doc_ids = doc_ids.to(device, dtype = torch.long)
        sent_mask = sent_mask.to(device, dtype = torch.long)
        doc_mask = doc_mask.to(device, dtype = torch.long)
        targets = targets.to(device, dtype = torch.float)

        outputs = model(sent_ids, doc_ids, sent_mask, doc_mask)

        lfn = nn.BCELoss()
        batch_loss = lfn(outputs, targets)
        total_loss_train += batch_loss.item() 

        n_correct = torch.count_nonzero(targets == (outputs > 0.5)).item()
        total_acc_train += n_correct
        nb_tr_examples+=targets.size(0)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    return total_acc_train / nb_tr_examples, total_loss_train / len(data_loader) 

def eval_fn(data_loader, model, device):
    """
    This function is utilize to evaluate the BERT architecture for the validation sample and 
    also calculate the loss for this epoch, store the relevant loss and at the end return 
    averaged out accuracy and loss for the validation sample

    args: 
    - data_loader: pytorch data loader object for the validation sample
    - model: trained BERT model which will be validated for validating answer start and end logits
    - device: pytorch device object to consider for validation

    return:
    - total_acc_val / nb_tr_examples: averaged out accuracy for the entire validation sample
    - total_loss_val / len(data_loader): averaged out loss for the entire validation sample
    """
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    nb_tr_examples = 0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            sent_ids = data['sent_ids']
            doc_ids = data['doc_ids']
            sent_mask = data['sent_mask']
            doc_mask = data['doc_mask'] 
            targets = data['targets'] 

            sent_ids = sent_ids.to(device, dtype = torch.long)
            doc_ids = doc_ids.to(device, dtype = torch.long)
            sent_mask = sent_mask.to(device, dtype = torch.long)
            doc_mask = doc_mask.to(device, dtype = torch.long)
            targets = targets.to(device, dtype = torch.float)

            outputs = model(sent_ids, doc_ids, sent_mask, doc_mask)

            lfn = nn.BCELoss()
            batch_loss = lfn(outputs, targets)
            total_loss_val += batch_loss.item() 

            n_correct = torch.count_nonzero(targets == (outputs > 0.5)).item()
            total_acc_val += n_correct
            nb_tr_examples+=targets.size(0)
        return total_acc_val / nb_tr_examples, total_loss_val / len(data_loader)