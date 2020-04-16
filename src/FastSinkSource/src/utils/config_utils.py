
# Add the fss base path so these imports work from anywhere
import sys
import os
#fss_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#sys.path.insert(0,fss_dir)
from ..algorithms import runner
from ..plot import plot_utils
import itertools


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
        weight_str="", postfix="", **kwargs):
    """
    Get the results file name(s) for this algorithm

    *file_type*: The type of results file to get. Options are:
        'pred-scores', 'cv-Xfolds', 'loso'

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
        alg_name = plot_utils.ALG_NAMES.get(alg,alg)
        alg_name += eval_str
        if len(combos) > 1: 
            alg_name = alg_name + params_str
        alg_results_files[alg_name] = results_file
    return alg_results_files
