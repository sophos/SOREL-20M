# SoReL-20M
Sophos-ReversingLabs 20 Million dataset

The code included in this repository produced the baseline models available at `s3://sorel-20m/09-DEC-2020/baselines`

This code depends on the SOREL dataset available via Amazon S3 at `s3://sorel-20m/09-DEC-2020/processed-data/`; to train the lightGBM models you can use the npz files available at `s3://sorel-20m/09-DC-2020/lightGBM-features/` or use the scripts included here to extract the required files from the processed data.

If you use this code or this data in your own research, please cite our paper: "SOREL-20M: A Large Scale Benchmark Dataset for Malicious PE Detection
" found at https://arxiv.org/abs/2012.07634 using the following citation:
```
@misc{harang2020sorel20m,
      title={SOREL-20M: A Large Scale Benchmark Dataset for Malicious PE Detection}, 
      author={Richard Harang and Ethan M. Rudd},
      year={2020},
      eprint={2012.07634},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

# Requirements

Python 3.6+.  See `environment.yml` for additional package requirements.

# A note on dataset size

The full size of this dataset is approximately 8TB.  It is highly recommended that you only obtain the specific elements you need. Files larger than 1GB are noted below.

```
s3://sorel-20m
|   Terms and Conditions of Use.pdf -- the terms you agree to by using this data and code
|
+---baselines
|   +---checkpoints
|   |   +---FFNN - per-epoch checkpoints for 5 seeds of the feed-forward neural network
|   |   +---lightGBM - final trained lightGBM model for 5 seeds
|   |
|   +---results
|       |  ffnn_results.json - index file of results, required for plotting
|       |  lgbm_results.json - index file of results, required for plotting
|       |
|       +---FFNN
|       |   +---seed0-seed4 - individual seed results, ~1GB each
|       |
|       +---lightgbm
|           +---seed0-seed4 - individual seed results, ~1GB each
|
+---binaries
|      approximately 8TB of zlib compressed malware binaries
|
+---lightGBM-features
|      test-features.npz - array of test data for lightGBM; 37GB
|      train-features.npz - array of training data for lightGBM; 113GB
|      validation-features.npz - array of validation data for lightGBM; 22GB
|
+---processed-data
    |   meta.db - contains index, labels, tags, and counts for the data; 3.5GB
    |
    +---ember_features - LMDB directory with baseline features, ~72GB
    +---pe_metadata - LMDB directory with full metadata dumps, ~480GB
```

Note: values in the LMDB files are serialized via msgpack and compressed via zlib; the code below handles this extraction automatically, however you will need to decompress and deserialize by hand if you use your own code to handle the data.

Please see the file `./pe_full_metadata_example/32c37c352802fb20004fa14053ac13134f31aff747dc0a2962da2ea1ea894d74.json` for an example of the metadata contained in the pe_metadata lmdb database.

# Quickstart

The main scripts of interest are:
1. `train.py` for training deep learning or (on a machine with sufficient RAM) LightGBM models
2. `evaluate.py` for taking a pretrained model and producing a results csv
3. `plot.py` for plotting the results

All scripts have multiple commands, documented via --help

Once you have cloned the repository, enter the repository directory and create a conda environment:

```
cd SoReL-20M
conda env create -f environment.yml
conda activate sorel
```

Ensure that you have the SOREL processed data in a local directory.  Edit `config.py` to indicate the device to use (CPU or CUDA) as well as the dataset location and desired checkpoint directory.  The dataset location should point to the folder that contains the `meta.db` file.


*Please note*: the complete contents of processed-data require approximately 552 GB of disk space, the bulk of which is the PE metadata and not used in training the baseline models.  If you only wish to retrain the baseline models, then you will need only the following files (approximately 78GB in total): 

```
/meta.db
/ember_features/data.mdb
/ember_features/lock.mdb
```

The file `shas_missing_ember_features.json` within this repository contains a list of sha256 values that indicate samples for which no Ember v2 feature values could be extracted; it is _highly recommended_ that the location of this file be passed to `--remove_missing_features` parameter in `train.train_network`, `evaluate.evaluate_network`, and `evaluate.evaluate_lgb` to significantly speed up the data loading time. If is it not provided, you should specify `--remove_missing_features='scan'` which will scan all keys to check for and remove ones with missing features prior to building the dataloader; if the dataloader reaches a missing feature it will cause an error.

You can train a neural network model with the following (note that config.py values can be overridden via command line switches:
```
python train.py train_network --remove_missing_features=shas_missing_ember_features.json 
```

Assuming that the checkpoint has been written to /home/ubuntu/checkpoints/ and you wish to place the results.csv file in /home/ubuntu/results/0 you may produce a test set evaluation as follows:

```
python evaluate.py evaluate_network /home/ubuntu/results/0 /home/ubuntu/checkpoints/epoch_9.pt 
```

To enable plotting of multiple series, the `plot.plot_roc_distributions_for_tag` function requires a json file that maps the name for a particular run to the results.csv file for that run.  

```
# Re-plot baselines -- note that the below command assumes 
# that the baseline models at s3://sorel-20m/09-DEC-2020/baselines
# have been downloaded to the /baselines directory
python plot.py plot_roc_distribution_for_tag /baselines/results/ffnn_results.json ./ffnn_results.png
```

# Neural network training

While a GPU allows for faster training (10 epochs can be completed in approximately 90 minutes), this model can be also trained via CPU; the provided results were obtained via GPU on an Amazon g3.4xlarge EC2 instance starting with a "Deep Learning AMI (Ubuntu 16.04) Version 26.0 (ami-025ed45832b817a35)" and updating it as above.  In practice, disk I/O loading features from the feature database seems to be a rate-limiting step assuming a GPU is used, so running on a machine with multiple cores and using a drive with high IOPS is recommended.  Training the network requires approximately 12GB of RAM when trained via CPU, though it varies slightly with the number of cores.  It is also highly recommended to use the `--remove_missing_features=shas_missing_ember_features.json` option as this significantly improves loading time of the data.

Note: if you get an error message `RuntimeError: received 0 items of ancdata` this is typically caused by the limit on the maximum number of open files being too low; this may be increased via the `ulimit` command.  In some cases -- if you use a large number of parallel workers -- it may also be neccessary to increase shared memory.

The commands to train and evaluate a neural network model are

```
python train.py train_network
python evaluate.py evaluate_network
```

Use `--help` for either script to see details and options.  The model itself is given in `nets.PENetwork`

# LightGBM training

Due to the size of the dataset, training a boosted model is difficult.  We use lightGBM, which has relatively memory-efficient data handlers, allowing it to fit a model in-memory using approximately 175GB of RAM.  The lightGBM model provided in this repository was trained on an Amazon m5.24xlarge instance.  

The script `build_numpy_arrays_for_lightgbm.py` will take training/validation/testing datasets and split them into three .npz files in the specified data location that can then be used for training a LightGBM model.  Please note that these files will be extremely large (113GB, 23GB, and 38GB, respectively) using the provided Ember features.

Alternatively, you may use the pre-extracted npz files available at `s3://sorel-20m/09-DEC-2020/lightGBM-features/` which contain Ember features using the default time splits for training, validation, and testing.

The lightGBM model can be trained in much the same manner as the neural network

```
python train.py train_lightGBM --train_npz_file=/dataset/train-features.npz --validation_npz_file=/dataset/validation-features.npz --model_configuration_file=./lightgbm_config.json --checkpoint_dir=/dataset/baselines/checkpoints/lightGBM/run0/
```

Assuming that you've placed the S3 dataset in /dataset as suggested above, this command will perform a single evaluation run.
```
python evaluate.py evaluate_lgb /dataset/baselines/checkpoints/lightGBM/seed0/lightgbm.model /home/ubuntu/lightgbm_eval --remove_missing_features=./shas_missing_ember_features.json 
```

The script used to generate the numpy array files from the database are found in `generate_numpy_arrays_for_lightgbm.dump_data_to_numpy`.  Note that this script requires approximately as much memory as training the model; a m5.24xlarge or equivalent EC2 instance type is recommended.

# Copyright and License

Copyright 2020, Sophos Limited. All rights reserved.

'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
Sophos Limited and Sophos Group. All other product and company
names mentioned are trademarks or registered trademarks of their
respective owners.


Licensed under the Apache License, Version 2.0 (the "License");
you may not use these files except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
