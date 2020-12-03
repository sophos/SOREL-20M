# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.

import baker
from copy import deepcopy
import sys
import numpy as np
from config import validation_test_split, train_validation_split, db_path
from generators import get_generator


@baker.command
def dump_data_to_numpy(mode, output_file, workers=1, batchsize=1000, remove_missing_features='scan'):
    """
    Produce numpy files required for training lightgbm model from SQLite + LMDB database.

    :param mode: One of 'train', 'validation', or 'test' representing which set of the
        data to process to file. Splits are obtained based on timestamps in config.py
    :param output_file: The name of the output file to produce for the indicated split.
    :param workers: How many worker processes to use (default 1)
    :param batchsize: The batch size to use in collecting samples (default 1000)
    :param remove_missing_features: How to check for and remove missing features; see
        README.md for recommendations (default 'scan')
    """
    _generator = get_generator(path=db_path,
                               mode=mode,
                               batch_size=batchsize,
                               use_malicious_labels=True,
                               use_count_labels=False,
                               use_tag_labels=False,
                               num_workers = workers,
                               remove_missing_features=remove_missing_features,
                               shuffle=False)
    feature_array = []
    label_array = []
    for i, (features, labels) in enumerate(_generator):
        feature_array.append(deepcopy(features.numpy()))
        label_array.append(deepcopy(labels['malware'].numpy()))
        sys.stdout.write(f"\r{i} / {len(_generator)}")
        sys.stdout.flush()
    np.savez(output_file, feature_array, label_array)
    print(f"\nWrote output to {output_file}")


if __name__ == '__main__':
    baker.run()
