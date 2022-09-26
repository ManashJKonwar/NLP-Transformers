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

            output = model(sent_ids, doc_ids, sent_mask, doc_mask)

            lfn = nn.BCELoss()
            batch_loss = lfn(outputs, targets)
            total_loss_val += batch_loss.item() 

            n_correct = torch.count_nonzero(targets == (outputs > 0.5)).item()
            total_acc_val += n_correct
            nb_tr_examples+=targets.size(0)
        return total_acc_val / nb_tr_examples, total_loss_val / len(data_loader)