# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.


import baker
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import json

default_tags = ['adware_tag', 'flooder_tag', 'ransomware_tag',
                'dropper_tag', 'spyware_tag',
                'packed_tag', 'crypto_miner_tag',
                'file_infector_tag', 'installer_tag',
                'worm_tag', 'downloader_tag']
default_tag_colors = ['r', 'r', 'r', 'g', 'g', 'b', 'b', 'm', 'm', 'c', 'c']
default_tag_linestyles = [':', '--', '-.', ':', '--', ':', '--', ':', '--', ':', '--']

style_dict = {tag: (color, linestyle) for tag, color, linestyle in zip(default_tags,
                                                                       default_tag_colors,
                                                                       default_tag_linestyles)}

style_dict['malware'] = ('k', '-')


def collect_dataframes(run_id_to_filename_dictionary):
    loaded_dataframes = {}
    for k, v in run_id_to_filename_dictionary.items():
        loaded_dataframes[k] = pd.read_csv(v)
    return loaded_dataframes


def get_tprs_at_fpr(result_dataframe, key, target_fprs=None):
    """
    Estimate the True Positive Rate for a dataframe/key combination
    at specific False Positive Rates of interest.
    :param result_dataframe: a pandas dataframe
    :param key: the name of the result to get the curve for; if (e.g.) the key 'malware' is provided
    the dataframe is expected to have a column names `pred_malware` and `label_malware`
    :param target_fprs: The FPRs at which you wish to estimate the TPRs; None (uses default np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]) or a 1-d numpy array
    :return: target_fprs, the corresponsing TPRs
    """
    if target_fprs is None:
        target_fprs = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    fpr, tpr, thresholds = get_roc_curve(result_dataframe, key)
    return target_fprs, np.interp(target_fprs, fpr, tpr)


def get_roc_curve(result_dataframe, key):
    """
    Get the ROC curve for a single result in a dataframe
    :param result_dataframe: a dataframe
    :param key: the name of the result to get the curve for; if (e.g.) the key 'malware' is provided
    the dataframe is expected to have a column names `pred_malware` and `label_malware`
    :return: false positive rates, true positive rates, and thresholds (all np.arrays)
    """
    labels = result_dataframe['label_{}'.format(key)]
    predictions = result_dataframe['pred_{}'.format(key)]
    return roc_curve(labels, predictions)


def get_auc_score(result_dataframe, key):
    """
    Get the Area Under the Curve for the indicated key in the dataframe
    :param result_dataframe: a dataframe
    :param key: the name of the result to get the curve for; if (e.g.) the key 'malware' is provided
    the dataframe is expected to have a column names `pred_malware` and `label_malware`
    :return: the AUC for the ROC generated for the provided key
    """
    labels = result_dataframe['label_{}'.format(key)]
    predictions = result_dataframe['pred_{}'.format(key)]
    return roc_auc_score(labels, predictions)


def interpolate_rocs(id_to_roc_dictionary, eval_fpr_points=None):
    """
    This function takes several sets of ROC results and interpolates them to a common set of
    evaluation (FPR) values to allow for computing e.g. a mean ROC or pointwise variance of the curve
    across multiple model fittings.
    :param list_of_rocs: a list of results from get_roc_score (or sklearn.metrics.roc_curve) of the
    form [(fpr_1, tpr_1, threshold_1), (fpr_2, tpr_2, threshold_2)...]
    :param eval_fpr_points: the set of FPR values at which to interpolate the results; defaults to
    `np.logspace(-6, 0, 1000)`
    :return:
        eval_fpr_points  -- the set of common points to which TPRs have been interpolated
        interpolated_tprs -- an array with one row for each ROC provided, giving the interpolated TPR for that ROC at
    the corresponding column in eval_fpr_points
    """
    if eval_fpr_points is None:
        eval_fpr_points = np.logspace(-6, 0, 1000)

    interpolated_tprs = {}

    for k, (fpr, tpr, thresh) in id_to_roc_dictionary.items():
        interpolated_tprs[k] = np.interp(eval_fpr_points, fpr, tpr)

    return eval_fpr_points, interpolated_tprs


def plot_roc_with_confidence(id_to_dataframe_dictionary, key, filename, include_range=False, style=None, std_alpha=.2,
                             range_alpha=.1):
    """
    Compute the mean and standard deviation of the ROC curve from a sequence of results
    and plot it with shading.
    """
    if not len(id_to_dataframe_dictionary) > 1:
        raise ValueError("Need a minimum of 2 result sets to plot confidence region; found {}".format(
            len(id_to_dataframe_dictionary)
        ))
    if style is None:
        if key in style_dict:
            color, linestyle = style_dict[key]
        else:
            raise ValueError(
                "No default style information is available for key {}; please provide (linestyle, color)".format(key))
    else:
        linestyle, color = style
    id_to_roc_dictionary = {k: get_roc_curve(df, key) for k, df in id_to_dataframe_dictionary.items()}
    fpr_points, interpolated_tprs = interpolate_rocs(id_to_roc_dictionary)
    tpr_array = np.vstack([v for v in interpolated_tprs.values()])
    mean_tpr = tpr_array.mean(0)
    std_tpr = np.sqrt(tpr_array.var(0))

    aucs = np.array([get_auc_score(v, key) for v in id_to_dataframe_dictionary.values()])
    mean_auc = aucs.mean()
    min_auc = aucs.min()
    max_auc = aucs.max()
    std_auc = np.sqrt(aucs.var())

    plt.figure(figsize=(12, 12))
    plt.semilogx(fpr_points, mean_tpr, color + linestyle, linewidth=2.0,
                 label=f"{key}: {mean_auc:5.3f}$\pm${std_auc:5.3f} [{min_auc:5.3f}-{max_auc:5.3f}]")
    plt.fill_between(fpr_points, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=std_alpha)
    if include_range:
        plt.fill_between(fpr_points, tpr_array.min(0), tpr_array.max(0), color=color, alpha=range_alpha)
    plt.legend()
    plt.xlim(1e-6, 1.0)
    plt.ylim([0., 1.])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.savefig(filename)
    plt.clf()


def plot_tag_results(dataframe, filename):
    all_tag_rocs = {tag: get_roc_curve(dataframe, tag) for tag in default_tags}
    eval_fpr_pts, interpolated_rocs = interpolate_rocs(all_tag_rocs)

    plt.figure(figsize=(12, 12))
    for tag in default_tags:
        color, linestyle = style_dict[tag]
        auc = get_auc_score(dataframe, tag)
        plt.semilogx(eval_fpr_pts, interpolated_rocs[tag], color + linestyle, linewidth=2.0,
                     label=f"{tag}:{auc:5.3f}")
    plt.legend(loc='best')
    plt.xlim(1e-6, 1.0)
    plt.ylim([0., 1.])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.savefig(filename)
    plt.clf()


@baker.command
def plot_tag_result(results_file, output_filename):
    """
    Takes a result file from a feedforward neural network model that includes all
    tags, and produces multiple overlaid ROC plots for each tag individually.

    :param results_file: complete path to a results.csv file that contains the output of 
        a model run.  Note that the model must have been trained with --use_tag_labels=True
        and evaluated using --evaluate_tags=True
    :param output_filename: the name of the file in which ot save the resulting plot.
    """
    id_to_resultfile_dict = {'run': results_file}
    id_to_dataframe_dict = collect_dataframes(id_to_resultfile_dict)
    plot_tag_results(id_to_dataframe_dict['run'], output_filename)


@baker.command
def plot_roc_distribution_for_tag(run_to_filename_json, output_filename, tag_to_plot='malware', linestyle=None, color=None,
                                  include_range=False, std_alpha=.2, range_alpha=.1):
    """
    Compute the mean and standard deviation of the TPR at a range of FPRS (the ROC curve)
    over several sets of results for a given tag.  The run_to_filename_json file must have
    the following format:
    {"run_id_0": "/full/path/to/results.csv/for/run/0/results.csv",
     "run_id_1": "/full/path/to/results.csv/for/run/1/results.csv",
      ...
    }
    
    :param run_to_filename_json: A json file that contains a key-value map that links run IDs to
        the full path to a results file (including the file name)
    :param output_filename: The filename to save the resulting figure to
    :param tag_to_plot: the tag from the results to plot; defaults to "malware"
    :param linestyle: the linestyle to use in the plot (defaults to the tag value in 
        plot.style_dict)
    :param color: the color to use in the plot (defaults to the tag value in 
        plot.style_dict)
    :param include_range: plot the min/max value as well (default False)
    :param std_alpha: the alpha value for the shading for standard deviation range
        (default 0.2)
    :param range_alpha: the alpha value for the shading for range, if plotted
        (default 0.1)
    """
    id_to_resultfile_dict = json.load(open(run_to_filename_json, 'r'))
    id_to_dataframe_dict = collect_dataframes(id_to_resultfile_dict)
    if color is None or linestyle is None:
        if not (color is None and linestyle is None):
            raise ValueError("both color and linestyle should either be specified or None")
        style = None
    else:
        style = (color, linestyle)
    plot_roc_with_confidence(id_to_dataframe_dict, tag_to_plot, output_filename, include_range=include_range, style=style,
                             std_alpha=std_alpha, range_alpha=range_alpha)


if __name__ == '__main__':
    baker.run()
