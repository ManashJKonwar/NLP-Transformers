__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from utils import get_batch_predictions, exact_match, f1_score

def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1,t1)
    l2 = nn.BCEWithLogitsLoss()(o2,t2)
    return l1 + l2

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    average_em, average_f1 = 0.0, 0.0
    running_loss = 0.0
    for data in tqdm(data_loader, total=len(data_loader)):
        input_ids = data['input_ids']
        mask = data['mask']
        offset_mapping = data['offset_mapping']
        token_type_ids = data['token_type_ids']
        start_positions = data['start_positions']
        end_positions = data['end_positions']

        input_ids = input_ids.to(device, dtype = torch.long)
        mask = mask.to(device, dtype = torch.long)
        offset_mapping = offset_mapping.to(device, dtype = torch.long)
        token_type_ids = token_type_ids.to(device, dtype = torch.long)
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)

        optimizer.zero_grad()
        start_logits, end_logits = model(
            input_ids=input_ids, 
            mask=mask,
            offset_mapping=offset_mapping,
            token_type_ids=token_type_ids,
        )

        loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        # Calculating batch level em and f1 scores
        predicted_answers = get_batch_predictions(data, start_logits, end_logits)
        actual_answers = data['answers']
        em = np.mean(exact_match(predicted_answers, actual_answers))
        f1 = np.mean(f1_score(predicted_answers, actual_answers))
        average_em += em
        average_f1 += f1

    average_em = average_em / (len(data_loader))
    average_f1 = average_f1 / (len(data_loader))
    training_loss = running_loss / len(data_loader)
    return training_loss, average_em, average_f1

def eval_fn(data_loader, model, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            input_ids = data['input_ids']
            mask = data['mask']
            offset_mapping = data['offset_mapping']
            token_type_ids = data['token_type_ids']
            start_position = data['start_position']
            end_position = data['end_position']

            input_ids = input_ids.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)
            offset_mapping = offset_mapping.to(device, dtype = torch.long)
            token_type_ids = token_type_ids.to(device, dtype = torch.long)
            start_position = start_position.to(device)
            end_position = end_position.to(device)

            start_logits, end_logits = model(
                input_ids=input_ids, 
                mask=mask,
                offset_mapping=offset_mapping,
                token_type_ids=token_type_ids,
            )

            loss = loss_fn(start_logits, end_logits, start_position, end_position)
            loss.backward()
            running_loss += loss.item()
        
        valid_loss = running_loss/len(data_loader)
        return valid_loss