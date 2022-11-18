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

import config

from utils import get_batch_predictions, exact_match, f1_score

def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return (l1 + l2) / 2

def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    This function is utilize to train the BERT architecture for the training sample and 
    also calculate the loss for this epoch, store the relevant cross entropy loss and at the end return 
    averaged out loss, exact words match score, f1 score for the train sample considering the predicted 
    start and end logits for the answer phrase followed by actual start and end logits within the context

    args: 
    - data_loader: pytorch data loader object for the train sample
    - model: BERT model which will be trained for predicted start and end logits for answers to relevant questions
    - optimizer: Adam optimizer to adjust after each data pass
    - device: pytorch device object to consider for training
    - scheduler: schedule object with a learning rate that decreases linearly from the initial lr set in the optimizer to 0

    return:
    - average_em: averaged out exact word match for the entire train sample
    - average_f1: f1 score for predictions and actual answer logits
    - training_loss: cross entropy loss for predicted and actual answer logits
    """
    model.train()
    average_em, average_f1 = 0.0, 0.0
    running_loss = 0.0
    for data in tqdm(data_loader, total=len(data_loader)):
        input_ids = data['input_ids'].squeeze(1)
        mask = data['mask'].squeeze(1)
        token_type_ids = data['token_type_ids'].squeeze(1)
        context_start_idx = data['context_start_idx']
        context_end_idx = data['context_end_idx']
        start_positions = data['start_positions']
        end_positions = data['end_positions']
        targets_start = data['targets_start']
        targets_end = data['targets_end']

        input_ids = input_ids.to(device, dtype = torch.long)
        mask = mask.to(device, dtype = torch.long)
        token_type_ids = token_type_ids.to(device, dtype = torch.long)
        context_start_idx = context_start_idx.to(device)
        context_end_idx = context_end_idx.to(device)
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)
        targets_start = targets_start.to(device, dtype = torch.float)
        targets_end = targets_end.to(device, dtype = torch.float)

        optimizer.zero_grad()
        start_logits, end_logits = model(
            ids=input_ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(start_logits, end_logits, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        # Calculating batch level em and f1 scores
        predicted_answers = get_batch_predictions(data, start_logits, end_logits, batch_size=config.TRAIN_BATCH_SIZE)
        actual_answers = data['answers']
        em = np.mean(exact_match(predicted_answers, actual_answers))
        f1 = np.mean(f1_score(predicted_answers, actual_answers))
        average_em += em
        average_f1 += f1

    average_em = average_em / len(data_loader)
    average_f1 = average_f1 / len(data_loader)
    training_loss = running_loss / len(data_loader)
    return average_em, average_f1, training_loss

def eval_fn(data_loader, model, device):
    """
    This function is utilize to evaluate the BERT architecture for the validation sample and 
    also calculate the loss for this epoch, store the relevant loss and at the end return 
    averaged out loss, exact words match score, f1 score for the validation sample

    args: 
    - data_loader: pytorch data loader object for the validation sample
    - model: trained BERT model which will be validated for validating answer start and end logits
    - device: pytorch device object to consider for validation

    return:
    - average_em: averaged out exact word match for the entire validation sample
    - average_f1: f1 score for predictions and actual answer logits
    - training_loss: cross entropy loss for predicted and actual answer logits
    """
    model.eval()
    average_em, average_f1 = 0.0, 0.0
    running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            input_ids = data['input_ids'].squeeze(1)
            mask = data['mask'].squeeze(1)
            token_type_ids = data['token_type_ids'].squeeze(1)
            context_start_idx = data['context_start_idx']
            context_end_idx = data['context_end_idx']
            start_positions = data['start_positions']
            end_positions = data['end_positions']
            targets_start = data['targets_start']
            targets_end = data['targets_end']

            input_ids = input_ids.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)
            token_type_ids = token_type_ids.to(device, dtype = torch.long)
            context_start_idx = context_start_idx.to(device)
            context_end_idx = context_end_idx.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)
            targets_start = targets_start.to(device, dtype = torch.float)
            targets_end = targets_end.to(device, dtype = torch.float)

            start_logits, end_logits = model(
                ids=input_ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            loss = loss_fn(start_logits, end_logits, targets_start, targets_end)
            running_loss += loss.item()

            # Calculating batch level em and f1 scores
            predicted_answers = get_batch_predictions(data, start_logits, end_logits, batch_size=config.VALID_BATCH_SIZE)
            actual_answers = data['answers']
            em = np.mean(exact_match(predicted_answers, actual_answers))
            f1 = np.mean(f1_score(predicted_answers, actual_answers))
            average_em += em
            average_f1 += f1
        
        average_em = average_em / len(data_loader)
        average_f1 = average_f1 / len(data_loader)
        valid_loss = running_loss / len(data_loader)
        return average_em, average_f1, valid_loss