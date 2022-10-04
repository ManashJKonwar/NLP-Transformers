__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    total_em_train = 0
    total_f1_train = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        input_ids = data['ids']
        input_mask = data['mask']

        input_ids = input_ids.to(device, dtype = torch.long)
        input_mask = input_mask.to(device, dtype = torch.long)

        outputs = model(input_ids, input_mask)
    return total_em_train, total_f1_train

def eval_fn(data_loader, model, device):
    model.eval()
    total_em_val = 0
    total_f1_val = 0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            input_ids = data['ids']
            input_mask = data['mask']

            input_ids = input_ids.to(device, dtype = torch.long)
            input_mask = input_mask.to(device, dtype = torch.long)

            outputs = model(input_ids, input_mask)
        return total_em_val, total_f1_val