#This script is for designing a way to choose specific alpha value for RL and RWR algorithms
import os, sys
import yaml
import argparse
import numpy as np
from sympy import Point, Line, Segment
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
import src.scripts.utils as script_utils

from src.FastSinkSource.src.algorithms import rl_genemania as gm
from src.FastSinkSource.src.algorithms import PageRank_Matrix as rwr
from src.scripts.plot_utils import *

import time

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        # config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to analyze the top contributors of each prediction's "
                                                 "diffusion score, as well as the effective diffusion (i.e.,"
                                                 " fraction of diffusion received from non-neighbors)")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/config_files/provenance/provenance_biogrid_y2hsept22_go.yaml"
                       ,help="Configuration file used when running FSS. ")

    # group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
    #                     "fss_inputs/config_files/provenance/provenance_biogrid_y2hsept22_s12.yaml"
    #                    , help="Configuration file used when running FSS. ")

    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--only_loss_diff', type=bool, default=False)

    return parser


def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """

    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    sig_cutoff = kwargs.get('stat_sig_cutoff')
    sig_str = "-sig%s" % (str(sig_cutoff).replace('.', '_')) if sig_cutoff else ""

    for dataset in input_settings['datasets']:

        # Store data (serialize)
        filename = config_map['output_settings']['output_dir'] + \
                   "/viz/%s/%s/param_select/" % (
                       dataset['net_version'],
                       dataset['exp_name']) + 'loss_intersect.pickle'

        if (not kwargs.get('only_loss_diff') or (not os.path.isfile(filename))):

            print("Loading data for %s" % (dataset['net_version']))
            # load the network and the positive examples for each term
            net_obj, ann_obj, _ = setup_dataset(
                dataset, input_dir, **kwargs)
            prots, node2idx = net_obj.nodes, net_obj.node2idx

            #declare a dict of dict here:
            #level1 key = alg, level2 key = term, value = (min_alpha, min_beta, min_difference between two loss terms)
            loss_diff = {}
            for term in ann_obj.terms:
            # for term in np.array(['GO:0098656']):
                term_idx = ann_obj.term2idx[term]
                orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
                orig_pos = [prots[p] for p in orig_pos_idx]
                pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
                n_pos = len(pos_nodes_idx)

                for alg_name in alg_settings:
                    if alg_name not in loss_diff:
                        loss_diff[alg_name] = {}

                    if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                        # get the alpha values to use
                        alphas = alg_settings[alg_name]['alpha']

                        alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                            output_dir, dataset, alg_settings, [alg_name], **kwargs)

                        loss_term1_across_alphas = {}
                        loss_term2_across_alphas = {}
                        loss_term1_across_betas = {}
                        loss_term2_across_betas = {}

                        for alpha, alg in zip(alphas, alg_pred_files):
                            pred_file = alg_pred_files[alg]

                            # Now find term_spec prediction file
                            pred_file = script_utils.term_based_pred_file(pred_file, term)

                            if not os.path.isfile(pred_file):
                                print("Warning: %s not found. skipping" % (pred_file))
                                continue
                            # print("reading %s for alpha=%s" % (pred_file, alpha))
                            df = pd.read_csv(pred_file, sep='\t')

                            df['idx'] = df['prot'].astype(str).apply(lambda x:node2idx[x])
                            # for the idx we don't have
                            # predicted value as they were predicted to be zero(we don't save 0 pred score, also
                            # we save upto 4 decimal number). insert them in df now.
                            t1=time.time()
                            zero_idices = [i for i in set(range(len(prots))).difference(set(df['idx']))]
                            # print('t1: ' ,time.time()-t1)

                            zero_scores = np.zeros(len(zero_idices))
                            zero_prots = [prots[z] for z in zero_idices]
                            zero_df = pd.DataFrame({'#term': [term]*len(zero_idices),
                                                    'prot': zero_prots,
                                                    'score':zero_scores,
                                                    'idx': zero_idices})
                            df = pd.concat([df, zero_df], axis=0)



                            # now sort df according to network 'idx' of proteins
                            df.sort_values(by=['idx'], inplace=True)

                            # Here we set the true label for unknown nodes to be 0. confirm with
                            # Murali if it is the right approach. According to Jeff's RL code when num_neg_example==0,
                            # we leave the unknowns label to be 0.

                            if alg_name =='genemaniaplus':
                                df['true'] = df['prot'].astype(str).apply(lambda x: 1  if x in orig_pos else 0)
                                loss_term1, loss_term2 = \
                                gm.compute_two_loss_terms(df['true'].to_numpy(), df['score'].to_numpy(), alpha, net_obj.W)
                            elif alg_name =='rwr':
                                df['true'] = df['prot'].astype(str).apply(lambda x: 1 / n_pos if x in orig_pos else 0)
                                loss_term1, loss_term2 = \
                                    rwr.compute_two_loss_terms(df['true'].to_numpy(), df['score'].to_numpy(), alpha,
                                                                    net_obj.W)

                            loss_term1_across_alphas[alpha] = loss_term1
                            loss_term2_across_alphas[alpha] = loss_term2

                            loss_term1_across_betas[round(float(1 / (1 + alpha)), 2)] = loss_term1
                            loss_term2_across_betas[round(float(1 / (1 + alpha)), 2)] = loss_term2


                        min_alpha, min_beta  = \
                            find_loss_intersection(loss_term1_across_alphas, loss_term2_across_alphas)
                        loss_diff[alg_name][term] = (min_alpha, min_beta)


                        #plot for quadratic loss terms values for specific network, alg, term across different alpha
                        net_alg_settings = config_map['output_settings']['output_dir'] + \
                                           "/viz/%s/%s/param_select/%s/" % (
                                           dataset['net_version'], dataset['exp_name'], alg_name)

                        title = dataset['plot_exp_name'] + '_' + term + '_' + plot_alg_name(alg_name)
                        outfile_prefix = net_alg_settings + term +'_quad_loss_terms'
                        os.makedirs(os.path.dirname(outfile_prefix), exist_ok=True)
                        plot_loss_terms(loss_term1_across_alphas, loss_term2_across_alphas, 'Alpha',
                                        title, outfile_prefix+'_alpha.png')
                        if alg_name=='genemaniaplus':
                            plot_loss_terms(loss_term1_across_betas,loss_term2_across_betas, 'Beta', title, outfile_prefix+'_beta.png')
            with open(filename, 'wb') as handle:
                pickle.dump(loss_diff, handle, protocol=pickle.HIGHEST_PROTOCOL)


        else:
            # Load data (deserialize)
            with open(filename, 'rb') as handle:
                loss_diff = pickle.load(handle)

        ##save loss intersections as pickle

        for alg_name in loss_diff:
            title = dataset['plot_exp_name'] + '_' +dataset['exp_name']+'_'+ plot_alg_name(alg_name)
            outfile_prefix = config_map['output_settings']['output_dir'] + \
                            "/viz/%s/%s/param_select/%s/" % (
                            dataset['net_version'],
                            dataset['exp_name'], alg_name) + 'difference_btn_quad_loss_terms'
            plot_min_diff(loss_diff[alg_name], 'Alpha', title, outfile_prefix + '_alpha.png')

            if alg_name=='genemaniaplus':
                plot_min_diff(loss_diff[alg_name], 'beta', title, outfile_prefix + '_beta.png')





def find_loss_intersection(loss_term1_across_alphas, loss_term2_across_alphas):
    '''
    This function will output the alpha, beta where the two curve for the two loss functions intersect
    '''

    loss1 = list(loss_term1_across_alphas.values())
    loss2 = list(loss_term2_across_alphas.values())
    alphas = list(loss_term1_across_alphas.keys())
    total_alphas = len(alphas)

    for i in range(0,total_alphas-1,1):
        if (np.sign(loss1[i]-loss2[i]) * np.sign(loss1[i+1]-loss2[i+1])!=1):
            #that means intersection is in between alphas[i] and alphas[i+1]
            p1, p2, p3, p4 = Point(alphas[i], loss1[i]), Point(alphas[i+1], loss1[i+1]),\
                             Point(alphas[i], loss2[i]), Point(alphas[i+1], loss2[i+1])
            l1 = Segment(p1, p2)
            l2 = Segment(p3, p4)

            # using intersection() method
            intersection_point = l1.intersection(l2)
            #taking the first intersection point and also taking the x coordinate
            intersection_alpha = round(float(intersection_point[0][0]),2)
            print(intersection_alpha)
    intersection_beta = round(float(1 / (1 + intersection_alpha)), 2)
    return intersection_alpha, intersection_beta





def plot_loss_terms(loss_term1_across_param, loss_term2_across_param, xlabel,title,filename):
    loss_term1_across_param = dict(sorted(loss_term1_across_param.items()))
    loss_term2_across_param = dict(sorted(loss_term2_across_param.items()))

    params = list(loss_term1_across_param.keys())
    loss_term1 = list(loss_term1_across_param.values())
    loss_term2 = list(loss_term2_across_param.values())
    x_data = np.arange(0, len(loss_term1), 1)
    
    
    plt.plot(x_data, loss_term1, label = 'term1')
    plt.plot(x_data, loss_term2, label = 'term2')
    
    plt.xticks(x_data, params)
    plt.xlabel(xlabel)
    plt.ylabel('Quadratic loss terms')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_min_diff(loss_diff_per_alg, ylabel, title, filename):
    if ylabel=='Alpha':
        tuple_index = 0
    else:
        tuple_index = 1

    #min_param_dict contains the  alpha/beta value for for which difference in loss terms is minimum
    min_param_dict ={term: loss_diff_per_alg[term][tuple_index] for term in loss_diff_per_alg}
    # min_diff_dict ={term: loss_diff_per_alg[term][2] for term in loss_diff_per_alg}
    # min_param_dict = dict(sorted(min_param_dict.items()))
    min_param_dict=dict(sorted(min_param_dict.items(), key=lambda item: item[1]))
    terms = list(min_param_dict.keys())
    min_params = list(min_param_dict.values())

    x_data = np.arange(0, len(terms), 1)
    plt.bar(x_data, min_params)

    plt.xticks(x_data, terms, rotation=90)
    plt.xlabel('Terms')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
