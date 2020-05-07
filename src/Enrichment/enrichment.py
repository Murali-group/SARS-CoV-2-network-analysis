
import argparse
import yaml
from collections import defaultdict
import os
import sys
#from tqdm import tqdm
import copy
import time
#import numpy as np
#from scipy import sparse
import pandas as pd
#import subprocess

# packages in this repo
from src.utils import parse_utils as utils
from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        #config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    # TODO check to make sure the inputs are correct in config_map

    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to test for enrichment of the top predictions among given genesets")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, required=True,
                       help="Configuration file used when running FSS. " +
                       "Must have a 'genesets_to_test' section for this script. ")
    group.add_argument('--k-to-test', '-k', type=int, action="append",
                       help="k-value(s) for which to get the top-k predictions to test. " +
                       "If not specified, will check the config file. Default=100")
    group.add_argument('--range-k-to-test', '-K', type=int, nargs=3,
                       help="Specify 3 integers: starting k, ending k, and step size. " +
                       "If not specified, will check the config file.")

    group = parser.add_argument_group('FastSinkSource Pipeline Options')
    group.add_argument('--alg', '-A', dest='algs', type=str, action="append",
                       help="Algorithms for which to get results. Must be in the config file. " +
                       "If not specified, will get the list of algs with 'should_run' set to True in the config file")
    group.add_argument('--num-reps', type=int, 
                       help="Number of times negative sampling was repeated to compute the average scores. Default=1")
    group.add_argument('--sample-neg-examples-factor', type=float, 
                       help="Factor/ratio of negatives to positives used when making predictions. " +
                       "Not used for methods which use only positive examples.")
    group.add_argument('--force-run', action='store_true', default=False,
                       help="Force re-running the enrichment tests, and re-writing the output files")

#    # additional parameters
#    group = parser.add_argument_group('Additional options')
#    group.add_argument('--forcealg', action="store_true", default=False,
#            help="Force re-running algorithms if the output files already exist")
#    group.add_argument('--forcenet', action="store_true", default=False,
#            help="Force re-building network matrix from scratch")
#    group.add_argument('--verbose', action="store_true", default=False,
#            help="Print additional info about running times and such")

    return parser


def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)
    algs = config_utils.get_algs_to_run(alg_settings, **kwargs)
    del kwargs['algs']

    genesets_to_test = config_map.get('genesets_to_test')
    if genesets_to_test is None or len(genesets_to_test) == 0:
        print("ERROR: no genesets specified to test for overlap. " +
              "Please add them under 'genesets_to_test'. \nQuitting") 
        sys.exit()

    # first load the gene sets
    geneset_groups = {}
    for geneset_to_test in genesets_to_test:
        name = geneset_to_test['name'] 
        gmt_file = "%s/genesets/%s/%s" % (
            input_dir, name, geneset_to_test['gmt_file'])
        if not os.path.isfile(gmt_file):
            print("WARNING: %s not found. skipping" % (gmt_file))
            sys.exit()

        geneset_groups[name] = utils.parse_gmt_file(gmt_file)  

    # for each dataset, extract the path(s) to the prediction files,
    # read in the predictions, and test for the statistical significance of overlap 
    for dataset in input_settings['datasets']:
        print("Loading data for %s" % (dataset['net_version']))
        # load the network and the positive examples for each term
        net_obj, ann_obj, eval_ann_obj = run_eval_algs.setup_dataset(
            dataset, input_dir, alg_settings, **kwargs) 
        prots = net_obj.nodes
        print("\t%d total prots" % (len(prots)))
        # TODO using this for the SARS-CoV-2 project,
        # but this should really be a general purpose script
        # and to work on any number of terms 
        orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, 0)
        orig_pos = [prots[p] for p in orig_pos_idx]
        print("\t%d original positive examples" % (len(orig_pos)))
        #pos_neg_file = "%s/%s" % (input_dir, dataset['pos_neg_file'])
        #df = pd.read_csv(pos_neg_file, sep='\t')
        #orig_pos = df[df['2020-03-sarscov2-human-ppi'] == 1]['prots']
        #print("\t%d original positive examples" % (len(orig_pos)))

        # now load the predictions, test at the various k values, and TODO plot
        k_to_test = dataset['k_to_test'] if 'k_to_test' in dataset else kwargs.get('k_to_test', [])
        range_k_to_test = dataset['range_k_to_test'] if 'range_k_to_test' in dataset \
                          else kwargs.get('range_k_to_test')
        if range_k_to_test is not None:
            k_to_test += list(range(
                range_k_to_test[0], range_k_to_test[1], range_k_to_test[2]))
        # if nothing was set, use the default value
        if k_to_test is None or len(k_to_test) == 0:
            k_to_test = [100]
        print("\ttesting %d k values: %s" % (len(k_to_test), ", ".join([str(k) for k in k_to_test])))

        # now load the prediction scores
        dataset_name = config_utils.get_dataset_name(dataset) 
        alg_pred_files = config_utils.get_dataset_alg_prediction_files(
            output_dir, dataset, alg_settings, algs, **kwargs)
        for alg, pred_file in alg_pred_files.items():
            if not os.path.isfile(pred_file):
                print("Warning: %s not found. skipping" % (pred_file))
                continue
            print("reading: %s" % (pred_file))
            df = pd.read_csv(pred_file, sep='\t')
            # remove the original positives
            df = df[~df['prot'].isin(orig_pos)]
            df.reset_index(inplace=True, drop=True)
            df = df[['prot', 'score']]
            df.sort_values(by='score', ascending=False, inplace=True)
            print(df.head())

        print("TODO finish this script by testing for enrichment")


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
