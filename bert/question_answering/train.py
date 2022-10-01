__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
import json
import numpy as np
import pandas as pd

from sklearn import model_selection

import config
import dataset

def process_data(input_file_path, is_train=True, record_path = ['data','paragraphs','qas','answers'], verbose = 1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if os.path.exists(config.TRAINING_FILE) and is_train:
        df = pd.read_csv(config.TRAINING_FILE)
        return df
    elif os.path.exists(config.VALID_FILE) and not is_train:
        df = pd.read_csv(config.VALID_FILE)
        return df
    else:
        if verbose:
            print("Reading the json file")    
        file = json.loads(open(input_file_path).read())
        if verbose:
            print("processing...")
        # parsing different level's in the json file
        js = pd.json_normalize(file, record_path)
        m = pd.json_normalize(file, record_path[:-1])
        r = pd.json_normalize(file,record_path[:-2])
        
        if is_train:
            #combining it into single dataframe
            idx = np.repeat(r['context'].values, r.qas.str.len())
            ndx = np.repeat(m['id'].values,m['answers'].str.len())
            m['context'] = idx
            js['q_idx'] = ndx
            main = pd.concat([ m[['id','question','context']].set_index('id'),js.set_index('q_idx')],1,sort=False).reset_index()
            main['c_id'] = main['context'].factorize()[0]
            if verbose:
                print("shape of the dataframe is {}".format(main.shape))
                print("Done")

            main.to_csv(config.TRAINING_FILE, index=False) 
            return main
        else:
            #combining it into single dataframe
            idx = np.repeat(r['context'].values, r.qas.str.len())
            m['context'] = idx
            main = m[['id','question','context','answers']].set_index('id').reset_index()
            main['c_id'] = main['context'].factorize()[0]
            if verbose:
                print("shape of the dataframe is {}".format(main.shape))
                print("Done")
            
            main.to_csv(config.VALID_FILE, index=False)
            return main

if __name__ == "__main__":
    df_train = process_data(config.QnA_TRAINING_PATH)
    df_valid = process_data(config.QnA_VALIDATION_PATH, is_train=False)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.QuestionAnsweringDataset(data=df_train)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4, shuffle=True
    )