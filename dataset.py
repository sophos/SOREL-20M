# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.


from torch.utils import data
import lmdb
import sqlite3
import baker
import msgpack
import zlib
import numpy as np
import os
import tqdm
from logzero import logger

import config
import json

class LMDBReader(object):

    def __init__(self, path, postproc_func=None):
        self.env = lmdb.open(path, readonly=True, map_size=1e13, max_readers=1024)
        self.postproc_func = postproc_func

    def __call__(self, key):
        with self.env.begin() as txn:
            x = txn.get(key.encode('ascii'))
        if x is None:return None
        x = msgpack.loads(zlib.decompress(x),strict_map_key=False)
        if self.postproc_func is not None:
            x = self.postproc_func(x)
        return x


def features_postproc_func(x):
    x = np.asarray(x[0], dtype=np.float32)
    lz = x < 0
    gz = x > 0
    x[lz] = - np.log(1 - x[lz])
    x[gz] = np.log(1 + x[gz])
    return x


def tags_postproc_func(x):
    x = list(x[b'labels'].values())
    x = np.asarray(x)
    return x


class Dataset(data.Dataset):
    tags = ["adware", "flooder", "ransomware", "dropper", "spyware", "packed",
            "crypto_miner", "file_infector", "installer", "worm", "downloader"]

    def __init__(self, metadb_path, features_lmdb_path,
                 return_malicious=True, return_counts=True, return_tags=True, return_shas=False,
                 mode='train', binarize_tag_labels=True, n_samples=None, remove_missing_features=True,
                 postprocess_function=features_postproc_func):

        self.return_counts = return_counts
        self.return_tags = return_tags
        self.return_malicious = return_malicious
        self.return_shas = return_shas

        self.features_lmdb_reader = LMDBReader(features_lmdb_path, postproc_func=postprocess_function)


        retrieve = ["sha256"]
        if return_malicious:
            retrieve += ["is_malware"]
        if return_counts:
            retrieve += ["rl_ls_const_positives"]
        if return_tags:
            retrieve.extend(Dataset.tags)

        conn = sqlite3.connect(metadb_path)
        cur = conn.cursor()
        query = 'select ' + ','.join(retrieve)
        query += " from meta"

        if mode == 'train':
            query += ' where(rl_fs_t <= {})'.format(config.train_validation_split)
        elif mode == 'validation':
            query += ' where((rl_fs_t >= {}) and (rl_fs_t < {}))'.format(config.train_validation_split,
                                                                         config.validation_test_split)
        elif mode == 'test':
            query += ' where(rl_fs_t >= {})'.format(config.validation_test_split)
        else:
            raise ValueError('invalid mode: {}'.format(mode))

        logger.info('Opening Dataset at {} in {} mode.'.format(metadb_path, mode))

        if type(n_samples) != type(None):
            query += ' limit {}'.format(n_samples)
        vals = cur.execute(query).fetchall()
        conn.close()
        logger.info(f"{len(vals)} samples loaded.")
        # map the items we're retrieving to an index
        retrieve_ind = dict(zip(retrieve, list(range(len(retrieve)))))

        if remove_missing_features=='scan':
            logger.info("Removing samples with missing features...")
            indexes_to_remove = []
            logger.info("Checking dataset for keys with missing features.")
            temp_env = lmdb.open(features_lmdb_path, readonly=True, map_size=1e13, max_readers=256)
            with temp_env.begin() as txn:
                for index, item in tqdm.tqdm(enumerate(vals), total=len(vals), mininterval=.5, smoothing=0.):
                    if txn.get(item[retrieve_ind['sha256']].encode('ascii')) is None:
                        indexes_to_remove.append(index)
            indexes_to_remove = set(indexes_to_remove)
            vals = [value for index, value in enumerate(vals) if index not in indexes_to_remove]
            logger.info(f"{len(indexes_to_remove)} samples had no associated feature and were removed.")
            logger.info(f"Dataset now has {len(vals)} samples.")
        elif (remove_missing_features is False) or (remove_missing_features is None):
            pass
        else:
            # assume filepath
            logger.info(f"Trying to load shas to ignore from {remove_missing_features}...")
            with open(remove_missing_features, 'r') as f:
                shas_to_remove = json.load(f)
            shas_to_remove = set(shas_to_remove)
            vals = [value for value in vals if value[retrieve_ind['sha256']] not in shas_to_remove]
            logger.info(f"Dataset now has {len(vals)} samples.")
        self.keylist = list(map(lambda x: x[retrieve_ind['sha256']], vals))
        if self.return_malicious:
            self.labels = list(map(lambda x: x[retrieve_ind['is_malware']], vals))
        if self.return_counts:
            self.count_labels = list(map(lambda x: x[retrieve_ind['rl_ls_const_positives']], vals))
        if self.return_tags:
            self.tag_labels = np.asarray([list(map(lambda x: x[retrieve_ind[t]], vals)) for t in Dataset.tags]).T
            if binarize_tag_labels:
                self.tag_labels = (self.tag_labels != 0).astype(int)



    def __len__(self):
        return len(self.keylist)

    def __getitem__(self, index):
        labels = {}
        key = self.keylist[index]
        features = self.features_lmdb_reader(key)
        if self.return_malicious:
            labels['malware'] = self.labels[index]
        if self.return_counts:
            labels['count'] = self.count_labels[index]
        if self.return_tags:
            labels['tags'] = self.tag_labels[index]
        if self.return_shas:
            return key, features, labels
        else:
            return features, labels



if __name__ == '__main__':
    baker.run()
