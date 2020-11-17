# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.


from dataset import Dataset
import os
from torch.utils import data
import config
from multiprocessing import cpu_count

max_workers = cpu_count()


class GeneratorFactory(object):
    def __init__(self, ds_root, batch_size=None, mode='train', num_workers=max_workers, use_malicious_labels=False,
                 use_count_labels=False, use_tag_labels=False, return_shas=False, features_lmdb='ember_features',
                 remove_missing_features='scan', shuffle=None):
        if mode not in {'train', 'validation', 'test'}:
            raise ValueError('invalid mode {}'.format(mode))
        ds = Dataset(metadb_path=os.path.join(ds_root, 'meta.db'),
                     features_lmdb_path=os.path.join(ds_root, features_lmdb),
                     return_malicious=use_malicious_labels,
                     return_counts=use_count_labels,
                     return_tags=use_tag_labels,
                     return_shas=return_shas, mode=mode,
                     remove_missing_features=remove_missing_features)
        if batch_size is None:
            batch_size = 1024
        
        # check passed in value for shuffle; pick a good one if it's None
        if shuffle is not None:
            if not ( (shuffle is True) or (shuffle is False)):
                raise ValueError(f"'shuffle' should be either True or False, got {shuffle}")
        else:
            if mode=='train':shuffle=True
            else:shuffle=False

        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  'num_workers': num_workers}

        self.generator = data.DataLoader(ds, **params)

    def __call__(self):
        return self.generator


def get_generator(mode, path=config.db_path, use_malicious_labels=True, use_count_labels=True,
                  use_tag_labels=True,
                  batch_size=config.batch_size, return_shas=False,
                  remove_missing_features='scan', num_workers=None, shuffle=None, 
                  feature_lmdb = 'ember_features'):
    if num_workers is None:
        num_workers = max_workers
    return GeneratorFactory(path, batch_size=batch_size, mode=mode, num_workers=num_workers,
                            use_malicious_labels=use_malicious_labels,
                            use_count_labels=use_count_labels, use_tag_labels=use_tag_labels,
                            return_shas=return_shas, remove_missing_features=remove_missing_features,
                            shuffle=shuffle, features_lmdb=feature_lmdb)()

