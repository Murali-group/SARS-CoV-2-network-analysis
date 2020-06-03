
# Add the fss base path so these imports work from anywhere
import sys
import os
#fss_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#sys.path.insert(0,fss_dir)
from ..algorithms import runner
from ..plot import plot_utils
import itertools
import pandas as pd
import numpy as np
import yaml


def load_config_file(config_file):
    with open(config_file, 'r') as conf:
        #config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    return config_map


def setup_config_variables(config_map, **kwargs):
    """
    Function to setup the various args specified in kwargs
    """
    input_settings = config_map['input_settings']
    input_dir = input_settings['input_dir']
    alg_settings = config_map['algs']
    output_settings = config_map['output_settings']
    output_dir = output_settings['output_dir']
    # update the settings specified in this script with those set in the yaml file
    # TODO these should be called 'script_settings'
    if config_map.get('eval_settings'):
        for key, val in config_map['eval_settings'].items():
            # if the user provided a value for this parameter (i.e., in kwargs), don't overwrite it
            if key not in kwargs or kwargs[key] is None or kwargs[key] == "":
                kwargs[key] = val
    return input_settings, input_dir, output_dir, alg_settings, kwargs


def get_algs_to_run(alg_settings, algs=None, **kwargs):
    # if there aren't any algs specified by the command line (i.e., kwargs),
    # then use whatever is in the config file
    if algs is None:
        # these are the algs to run
        algs = []
        for alg in alg_settings:
            if alg_settings[alg].get('should_run', [True])[0] is True:
                algs.append(alg.lower())
    else:
        # make the alg names lower so capitalization won't make a difference
        algs = [a.lower() for a in algs]
        for alg in algs:
            if alg not in alg_settings:
                print("ERROR: alg '%s' not found in config file.\nQuitting" % (alg))
                sys.exit()
    return algs


def get_all_prediction_files(input_settings, output_dir, alg_settings, algs=None, **kwargs):
    """
    For each dataset and each algorithm, get the path to the file(s) containing the prediction scores
    """
    all_dataset_alg_pred_files = {}
    algs = get_algs_to_run(alg_settings, algs=algs)
    for dataset in input_settings['datasets']:
        dataset_name = get_dataset_name(dataset) 
        dataset_alg_pred_files = get_dataset_alg_prediction_files(
            output_dir, dataset, alg_settings, algs, **kwargs)
        all_dataset_alg_pred_files[dataset_name] = dataset_alg_pred_files
    return dataset_alg_pred_files


def get_dataset_name(dataset):
    dataset_name = "%s %s" % (dataset['net_version'], dataset['exp_name'])
    # if a name is given this experiment, then use that
    if 'plot_exp_name' in dataset:
        dataset_name = dataset['plot_exp_name']
    dataset['plot_exp_name'] = dataset_name
    return dataset_name


def get_dataset_alg_prediction_files(
        output_dir, dataset, alg_settings, algs, **kwargs):
    """
    Get the file paths to the results (e.g., prediction scores or cross-validation measures) files
    """
    dataset_alg_pred_files = {} 
    results_dir = "%s/%s/%s/" % (output_dir, dataset['net_version'], dataset['exp_name'])
    # this contains the weighting method used for the network if multiple networks were used (e.g., 'swsn' or 'gmw')
    weight_str = runner.get_weight_str(dataset)
    for alg in algs:
        alg_pred_files = get_alg_results_files(
            alg, alg_settings[alg], weight_str=weight_str, **kwargs)
        # add the full file path to the prediction score files
        alg_pred_files = {a: "%s/%s/%s" % (results_dir, alg, f) for a,f in alg_pred_files.items()}
        dataset_alg_pred_files.update(alg_pred_files)
    return dataset_alg_pred_files


def get_alg_results_files(
        alg, alg_params, file_type="pred-scores",
        weight_str="", postfix="", use_alg_plot_name=True, **kwargs):
    """
    Get the results file name(s) for this algorithm

    *file_type*: The type of results file to get. Options are:
        'pred-scores', 'cv-Xfolds', 'loso'
    *use_alg_plot_name*: Option to use the name of the algorithm used when plotting vs the standard full name. 
        e.g., GM+ vs genemaniaplus

    *returns*: a dictionary of algorithm name: file path
        algorithm name will include the settings used to run it if alg_params contains multiple combinations of parameters
    """
    alg_results_files = {}
    # generate all combinations of parameters specified
    combos = [dict(zip(alg_params.keys(), val))
        for val in itertools.product(
            *(alg_params[param] for param in alg_params))]
    for param_combo in combos:
        # first get the parameter string for this runner
        params_str = runner.get_runner_params_str(alg, param_combo, weight_str=weight_str, **kwargs)
        eval_str = runner.get_eval_str(alg, **kwargs)
        results_file = "%s%s%s.txt" % (file_type, params_str, postfix)
        if use_alg_plot_name is True:
            alg_name = plot_utils.ALG_NAMES.get(alg,alg)
            alg_name += eval_str
        else:
            alg_name = alg
        if len(combos) > 1: 
            alg_name = alg_name + params_str
        alg_results_files[alg_name] = results_file
    return alg_results_files


def get_pvals_apply_cutoff(df, pred_file, apply_cutoff=True, **kwargs):
    df.set_index('prot', inplace=True)
    stat_sig_params_str = "%srand-%s%s" % (
        kwargs.get('num_random_sets',1000), kwargs.get('num_bins',10),
        kwargs.get('bin_method','kmeans'))
    pval_file = pred_file.replace("networks/", "viz/networks/") \
        .replace(".txt", "-%s-pvals.tsv" % (stat_sig_params_str))
    if not os.path.isfile(pval_file):
        print("WARNING: %s does not exist. Leaving pval column empty" % (pval_file))
        df['pval'] = np.nan
        return df
    print("reading pvals from %s" % (pval_file))
    pval_df = pd.read_csv(pval_file, sep='\t', index_col=0)
    df['pval'] = pval_df['pval']
    topk = list(df.iloc[:300].index)
    df2 = df[df['pval'] < kwargs['stat_sig_cutoff']]
    topk2 = list(df2.iloc[:300].index)

    print("\t%d nodes with pval < %s among top 300" % (len(set(topk) & set(topk2)), kwargs['stat_sig_cutoff']))
    df2.reset_index(inplace=True, col_fill='prot')
    df2.reset_index(inplace=True, drop=True)
    #df = df[['prot', 'score']]
    df2.sort_values(by='score', ascending=False, inplace=True)
    if apply_cutoff:
        print(df2.head())
        return df2
    else:
        return df


