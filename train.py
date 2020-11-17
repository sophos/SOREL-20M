# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.


from dataset import Dataset
from nets import PENetwork
import warnings
import os
import baker
import torch
import torch.nn.functional as F
from torch.utils import data
import sys
from generators import get_generator
from config import device
import config
from logzero import logger
from copy import deepcopy
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import pickle
import json
import lightgbm as lgb


def compute_loss(predictions, labels, loss_wts={'malware': 1.0, 'count': 0.1, 'tags': 0.1}):
    """
    Compute losses for a malware feed-forward neural network (optionally with SMART tags 
    and vendor detection count auxiliary losses).

    :param predictions: a dictionary of results from a PENetwork model
    :param labels: a dictionary of labels 
    :param loss_wts: weights to assign to each head of the network (if it exists); defaults to 
        values used in the ALOHA paper (1.0 for malware, 0.1 for count and each tag)
    """
    loss_dict = {'total':0.}
    if 'malware' in labels:
        malware_labels = labels['malware'].float().to(device)
        malware_loss = F.binary_cross_entropy(predictions['malware'].reshape(malware_labels.shape), malware_labels)
        weight = loss_wts['malware'] if 'malware' in loss_wts else 1.0
        loss_dict['malware'] = deepcopy(malware_loss.item())
        loss_dict['total'] += malware_loss * weight
    if 'count' in labels:
        count_labels = labels['count'].float().to(device)
        count_loss = torch.nn.PoissonNLLLoss()(predictions['count'].reshape(count_labels.shape), count_labels)
        weight = loss_wts['count'] if 'count' in loss_wts else 1.0
        loss_dict['count'] = deepcopy(count_loss.item())
        loss_dict['total'] += count_loss * weight
    if 'tags' in labels:
        tag_labels = labels['tags'].float().to(device)
        tags_loss = F.binary_cross_entropy(predictions['tags'], tag_labels)
        weight = loss_wts['tags'] if 'tags' in loss_wts else 1.0
        loss_dict['tags'] = deepcopy(tags_loss.item())
        loss_dict['total'] += tags_loss * weight
    return loss_dict


@baker.command
def train_network(train_db_path=config.db_path,
                  checkpoint_dir=config.checkpoint_dir,
                  max_epochs=10,
                  use_malicious_labels=True,
                  use_count_labels=True,
                  use_tag_labels=True,
                  feature_dimension=2381,
                  random_seed=None, 
                  workers = None,
                  remove_missing_features='scan'):
    """
    Train a feed-forward neural network on EMBER 2.0 features, optionally with additional targets as
    described in the ALOHA paper (https://arxiv.org/abs/1903.05700).  SMART tags based on
    (https://arxiv.org/abs/1905.06262)
    

    :param train_db_path: Path in which the meta.db is stored; defaults to the value specified in `config.py`
    :param checkpoint_dir: Directory in which to save model checkpoints; WARNING -- this will overwrite any existing checkpoints without warning.
    :param max_epochs: How many epochs to train for; defaults to 10
    :param use_malicious_labels: Whether or not to use malware/benignware labels as a target; defaults to True
    :param use_count_labels: Whether or not to use the counts as an additional target; defaults to True
    :param use_tag_labels: Whether or not to use SMART tags as additional targets; defaults to True
    :param feature_dimension: The input dimension of the model; defaults to 2381 (EMBER 2.0 feature size)
    :param random_seed: if provided, seed random number generation with this value (defaults None, no seeding)
    :param workers: How many worker processes should the dataloader use (default None, use multiprocessing.cpu_count())
    :param remove_missing_features: Strategy for removing missing samples, with meta.db entries but no associated features,
        from the data (e.g. feature extraction failures).  
        Must be one of: 'scan', 'none', or path to a missing keys file.  
        Setting to 'scan' (default) will check all entries in the LMDB and remove any keys that are missing -- safe but slow. 
        Setting to 'none' will not perform a check, but may lead to a run failure if any features are missing.  Setting to
        a path will attempt to load a json-serialized list of SHA256 values from the specified file, indicating which
        keys are missing and should be removed from the dataloader.
    """
    workers = workers if workers is None else int(workers)
    os.system('mkdir -p {}'.format(checkpoint_dir))
    if random_seed is not None:
        logger.info(f"Setting random seed to {int(random_seed)}.")
        torch.manual_seed(int(random_seed))
    logger.info('...instantiating network')
    model = PENetwork(use_malware=True, use_counts=True, use_tags=True, n_tags=len(Dataset.tags),
                      feature_dimension=feature_dimension).to(device)
    opt = torch.optim.Adam(model.parameters())
    generator = get_generator(path=train_db_path,
                              mode='train',
                              use_malicious_labels=use_malicious_labels,
                              use_count_labels=use_count_labels,
                              use_tag_labels=use_tag_labels,
                              num_workers = workers,
                              remove_missing_features=remove_missing_features)
    val_generator = get_generator(path = train_db_path,
                                  mode='validation', 
                                  use_malicious_labels=use_malicious_labels,
                                  use_count_labels=use_count_labels,
                                  use_tag_labels=use_tag_labels,
                                  num_workers=workers,
                                  remove_missing_features=remove_missing_features)
    steps_per_epoch = len(generator)
    val_steps_per_epoch = len(val_generator)
    for epoch in range(1, max_epochs + 1):
        loss_histories = defaultdict(list)
        model.train()
        for i, (features, labels) in enumerate(generator):
            opt.zero_grad()
            features = deepcopy(features).to(device)
            out = model(features)
            loss_dict = compute_loss(out, deepcopy(labels))
            loss = loss_dict['total']
            loss.backward()
            opt.step()
            for k in loss_dict.keys():
                if k == 'total': loss_histories[k].append(deepcopy(loss_dict[k].detach().cpu().item()))
                else: loss_histories[k].append(loss_dict[k])
            loss_str = " ".join([f"{key} loss:{value:7.3f}" for key, value in loss_dict.items()])
            loss_str += " | "
            loss_str += " ".join([f"{key} mean:{np.mean(value):7.3f}" for key, value in loss_histories.items()])
            sys.stdout.write('\r Epoch: {}/{} {}/{} '.format(epoch, max_epochs, i + 1, steps_per_epoch) + loss_str)
            sys.stdout.flush()
            del features, labels # do our best to avoid weird references that lead to generator errors
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "epoch_{}.pt".format(str(epoch))))
        print()
        loss_histories = defaultdict(list)
        model.eval()
        for i, (features, labels) in enumerate(val_generator):
            features = deepcopy(features).to(device)
            with torch.no_grad():
                out = model(features)
            loss_dict = compute_loss(out, deepcopy(labels))
            loss = loss_dict['total']
            for k in loss_dict.keys():
                if k == 'total': loss_histories[k].append(deepcopy(loss_dict[k].detach().cpu().item()))
                else: loss_histories[k].append(loss_dict[k])
            loss_str = " ".join([f"{key} loss:{value:7.3f}" for key, value in loss_dict.items()])
            loss_str += " | "
            loss_str += " ".join([f"{key} mean:{np.mean(value):7.3f}" for key, value in loss_histories.items()])
            sys.stdout.write('\r   Val: {}/{} {}/{} '.format(epoch, max_epochs, i + 1, val_steps_per_epoch) + loss_str)
            sys.stdout.flush()
            del features, labels # do our best to avoid weird references that lead to generator errors
        print() 
    print('...done')



@baker.command
def train_lightGBM(train_npz_file, validation_npz_file, model_configuration_file, checkpoint_dir,
                   random_seed=None):
    """
    Train a lightGBM model.  Note that this is done entirely in-memory and requires a substantial 
    amount of RAM (approximately 175GB).  Baseline models were trained on an Amazon m5.24xlarge instance.

    :param train_npz_file: path to a .npz file containing featres in 'arr_0' and labels in 'arr_1' for the training data
    :param validation_npz_file: path to a .npz file containing featres in 'arr_0' and labels in 'arr_1' for the validation data
    :param model_configuration_file: path to a json file specifying lightGBM parameters (see lightgbm_config.json for an example)
    :param checkpoint_dir: location to write the trained model to
    :param random_seed: defaults to None (no seeding) otherwise an integer providing a fixed random seed for the experiment.
    """
    logger.info("Loading model config json file...")
    config = json.load(open(model_configuration_file, 'r'))
    if random_seed is not None:
        random_seed = int(random_seed)
        config['seed']=random_seed
        config['bagging_seed']=random_seed
        config['feature_fraction_seed']=random_seed
    logger.info("Loading train data...")
    train_npz = np.load(train_npz_file)
    train_fts, train_lbls = train_npz['arr_0'], train_npz['arr_1']
    val_npz = np.load(validation_npz_file)
    val_fts, val_lbls = val_npz['arr_0'], val_npz['arr_1']
    logger.info("Converting data to lightGMB.Dataset")
    train_data = lgb.Dataset(train_fts, label=train_lbls)
    val_data = lgb.Dataset(val_fts, label=val_lbls)
    logger.info("Starting training")

    bst = lgb.train(params=config, train_set=train_data, valid_sets=[val_data])

    os.system('mkdir -p {}'.format(checkpoint_dir))
    modelfile = os.path.join(checkpoint_dir, 'lightgbm.model')
    logger.info(f"Saving model to {modelfile}")
    bst.save_model(modelfile)



if __name__ == '__main__': 
    baker.run()
