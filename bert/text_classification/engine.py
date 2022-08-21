__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import torch.nn as nn
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    total_acc_train = 0
    total_loss_train = 0
    for train_ip, train_label in tqdm(data_loader, total=len(data_loader)):
        train_label = train_label.to(device)
        mask = train_ip['attention_mask'].to(device)
        input_id = train_ip['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)

        lfn = nn.CrossEntropyLoss()
        batch_loss = lfn(output, train_label.long())
        total_loss_train += batch_loss.item()

        acc = (output.argmax(dim=1) == train_label).sum().item()
        total_acc_train += acc

        model.zero_grad()
        batch_loss.backward()
        optimizer.step()
    return total_acc_train / len(data_loader), total_loss_train / len(data_loader)

def eval_fn(data_loader, model, device):
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    for val_ip, val_label in tqdm(data_loader, total=len(data_loader)):
        val_label = val_label.to(device)
        mask = val_input['attention_mask'].to(device)
        input_id = val_input['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)

        lfn = nn.CrossEntropyLoss()
        batch_loss = lfn(output, val_label.long())
        total_loss_val += batch_loss.item()
        
        acc = (output.argmax(dim=1) == val_label).sum().item()
        total_acc_val += acc
    return total_acc_val / len(data_loader), total_loss_val/ len(data_loader)