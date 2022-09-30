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

import config

def process_data(input_file_path, record_path = ['data','paragraphs','qas','answers'], verbose = 1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if os.path.exists(config.TRAINING_FILE):
        df = pd.read_csv(config.TRAINING_FILE)
        return df
    else:
        if verbose:
            print("Reading the json file")    
        file = json.loads(open(input_file_path).read())
        if verbose:
            print("processing...")
        # parsing different level's in the json file
        js = pd.io.json.json_normalize(file , record_path )
        m = pd.io.json.json_normalize(file, record_path[:-1] )
        r = pd.io.json.json_normalize(file,record_path[:-2])
        
        #combining it into single dataframe
        idx = np.repeat(r['context'].values, r.qas.str.len())
        ndx  = np.repeat(m['id'].values,m['answers'].str.len())
        m['context'] = idx
        js['q_idx'] = ndx
        main = pd.concat([ m[['id','question','context']].set_index('id'),js.set_index('q_idx')],1,sort=False).reset_index()
        main['c_id'] = main['context'].factorize()[0]
        if verbose:
            print("shape of the dataframe is {}".format(main.shape))
            print("Done")

        main.to_csv(config.TRAINING_FILE, index=False) 
        return main

if __name__ == "__main__":
    df_qna = process_data(config.QnA_TRAINING_PATH)