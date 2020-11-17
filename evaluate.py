# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.


import torch
import baker
from nets import PENetwork
from generators import get_generator
import tqdm
import os
from config import device
import config
from dataset import Dataset
import pickle
from logzero import logger
from copy import deepcopy
import pandas as pd
import numpy as np

all_tags = Dataset.tags

def detach_and_copy_array(array):
    if isinstance(array, torch.Tensor):
        return deepcopy(array.cpu().detach().numpy()).ravel()
    elif isinstance(array, np.ndarray):
        return deepcopy(array).ravel()
    else:
        raise ValueError("Got array of unknown type {}".format(type(array)))

def normalize_results(labels_dict, results_dict, use_malware=True, use_count=True, use_tags=True):
    """
    Take a set of results dicts and break them out into
    a single dict of 1d arrays with appropriate column names
    that pandas can convert to a DataFrame.
    """
    # we do a lot of deepcopy stuff here to avoid a FD "leak" in the dataset generator
    # see here: https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
    rv = {}
    if use_malware:
        rv['label_malware'] = detach_and_copy_array(labels_dict['malware'])
        rv['pred_malware'] = detach_and_copy_array(results_dict['malware'])
    if use_count:
        rv['label_count'] = detach_and_copy_array(labels_dict['count'])
        rv['pred_count'] = detach_and_copy_array(results_dict['count'])
    if use_tags:
        for column, tag in enumerate(all_tags):
            rv[f'label_{tag}_tag'] = detach_and_copy_array(labels_dict['tags'][:, column])
            rv[f'pred_{tag}_tag']=detach_and_copy_array(results_dict['tags'][:, column])
    return rv

@baker.command
def evaluate_network(results_dir, checkpoint_file,
                     db_path=config.db_path,
                     evaluate_malware=True,
                     evaluate_count=True,
                     evaluate_tags=True,
                     remove_missing_features='scan'):
    """
    Take a trained feedforward neural network model and output evaluation results to a csv in the specified location.

    :param results_dir: The directory to which to write the 'results.csv' file; WARNING -- this will overwrite any
        existing results in that location
    :param checkpoint_file: The checkpoint file containing the weights to evaluate
    :param db_path: the path to the directory containing the meta.db file; defaults to the value in config.py
    :param evaluate_malware: defaults to True; whether or not to record malware labels and predictions
    :param evaluate_count: defaults to True; whether or not to record count labels and predictions
    :param evaluate_tags: defaults to True; whether or not to record individual tag labels and predictions
    :param remove_missing_features: See help for remove_missing_features in train.py / train_network
    """
    os.system('mkdir -p {}'.format(results_dir))
    model = PENetwork(use_malware=True, use_counts=True, use_tags=True, n_tags=len(Dataset.tags),
                      feature_dimension=2381)
    model.load_state_dict(torch.load(checkpoint_file))
    model.to(device)
    generator = get_generator(mode='test', path=db_path, use_malicious_labels=evaluate_malware,
                              use_count_labels=evaluate_count,
                              use_tag_labels=evaluate_tags, return_shas=True,
                              remove_missing_features=remove_missing_features)
    logger.info('...running network evaluation')
    f = open(os.path.join(results_dir,'results.csv'),'w')
    first_batch = True
    for shas, features, labels in tqdm.tqdm(generator):
        features = features.to(device)
        predictions = model(features)
        results = normalize_results(labels, predictions)
        pd.DataFrame(results, index=shas).to_csv(f, header=first_batch)
        first_batch=False
    f.close()
    print('...done')


import lightgbm as lgb

@baker.command
def evaluate_lgb(lightgbm_model_file,
                 results_dir,
                 db_path=config.db_path,
                 remove_missing_features='scan'
                 ):
    """
    Take a trained lightGBM model and perform an evaluation on it. Results will be saved to 
    results.csv in the path specified in results_dir

    :param lightgbm_model_file: Full path to the trained lightGBM model
    :param results_dir: The directory to which to write the 'results.csv' file; WARNING -- this will overwrite any
        existing results in that location
    :param db_path: the path to the directory containing the meta.db file; defaults to the value in config.py
    :param remove_missing_features: See help for remove_missing_features in train.py / train_network
    """
    os.system('mkdir -p {}'.format(results_dir))

    logger.info(f'Loading lgb model from {lightgbm_model_file}')
    model = lgb.Booster(model_file=lightgbm_model_file)
    generator = get_generator(mode='test', path=db_path, use_malicious_labels=True,
                              use_count_labels=False,
                              use_tag_labels=False, return_shas=True,
                              remove_missing_features=remove_missing_features)
    logger.info('running lgb evaluation')
    f = open(os.path.join(results_dir, 'results.csv'), 'w')
    first_batch = True
    for shas, features, labels in tqdm.tqdm(generator):
        predictions = {'malware':model.predict(features)}
        results = normalize_results(labels, predictions, use_malware=True, use_count=False, use_tags=False)
        pd.DataFrame(results, index=shas).to_csv(f, header=first_batch)
        first_batch = False
    f.close()
    print('...done')

if __name__ == '__main__':
    baker.run()
