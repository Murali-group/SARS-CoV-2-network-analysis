import os, sys
import yaml
import argparse
import itertools
from tqdm import tqdm
from collections import defaultdict
from collections import Iterable
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy import sparse as sp
import networkx as nx
import matplotlib
if __name__ == "__main__":
    # Use this to save files remotely. 
    matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

# local imports
from src import setup_datasets
from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
from src.FastSinkSource.src.evaluate import stat_sig
from src.FastSinkSource.src.algorithms import genemania_runner as gm_runner
from src.FastSinkSource.src.algorithms import genemania as gm
from src.FastSinkSource.src.evaluate import stat_sig

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    config_map_list = []
    for config in args.config.split(","):
        with open(config, 'r') as conf:
            config_map_list.append(yaml.load(conf))

    return config_map_list, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to compare the diffusion values to each virus interactor from all other virus interactors with corresponding diffusion values to a set of random proteins.")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, required=True,
                       help="Configuration files used when running FSS. ")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    group.add_argument('--out-dir', type=str, default="rand_diffusion_cross_validation",
                       help="Directory in which to write output files.")
    group.add_argument('--num-random-sets', type=int, 
                       help="Number of random proteins in sets compared against other virus interactors.")

    return parser


def main(config_map_list, **kwargs):

  input_settings, input_dir, output_dir, alg_settings, kwargs \
      = config_utils.setup_config_variables(config_map_list[0], **kwargs)

  alphas = []
  alpha_settings = alg_settings['genemaniaplus']['alpha'][0]
  if isinstance(alpha_settings, Iterable):
    for a in alpha_settings:
      alphas.append(a)
  else:
    alphas.append(alpha_settings)

  for alpha in alphas:
    interactors = []
    interactor_values = []
    rand_values = []

    uniprot_to_gene = {}
    if kwargs.get('id_mapping_file'):
      print("Reading %s" % (kwargs['id_mapping_file']))
      df = pd.read_csv(kwargs['id_mapping_file'], sep='\t', header=0)
      ## keep only the first gene for each UniProt ID
      uniprot_to_gene = {p: genes.split(' ')[0] for p, genes in zip(df['Entry'], df['Gene names'].astype(str))}

    for config_map in config_map_list:
      """
      *config_map*: everything in the config file
      *kwargs*: all of the options passed into the script
      """
      # extract the general variables from the config map
      input_settings, input_dir, output_dir, alg_settings, kwargs \
          = config_utils.setup_config_variables(config_map, **kwargs)

      # for each dataset, extract the path(s) to the prediction files,
      # read in the predictions, and test for the statistical significance of overlap 
      for dataset in input_settings['datasets']:
          dataset_name = config_utils.get_dataset_name(dataset)
          print("Loading data for %s" % (dataset['net_version']))
          # load the network and the positive examples for each term
          net_obj, ann_obj, eval_ann_obj = run_eval_algs.setup_dataset(
              dataset, input_dir, **kwargs) 
          prots, node2idx = net_obj.nodes, net_obj.node2idx
          print("\t%d total prots" % (len(prots)))

          orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, 0)
          orig_pos = [prots[p] for p in orig_pos_idx]
          print("\t%d original positive examples" % (len(orig_pos)))
          # convert the virus interactor nodes to ids
          interactor_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]

          interactor_value, rand_value = eval_diffusion_cross_validation(dataset, net_obj, 
              interactor_nodes_idx, uniprot_to_gene, alpha, **kwargs)

          interactors.append(dataset['exp_name'])
          interactor_values.append(interactor_value)
          rand_values.append(rand_value)

    plot_diffusion_cross_validation(interactors, interactor_values, rand_values, alpha, **kwargs)

def eval_diffusion_cross_validation(dataset, net_obj, interactor_nodes_idx, uniprot_to_gene, alpha, **kwargs):
    num_random_sets = kwargs.get('num_random_sets', 1000)
    out_dir = kwargs.get('out_dir', 'rand_diffusion_cross_validation')
    table_file = out_dir + "/%s-alpha-%.2g-rand-diffusion-cross-validation.tsv" % (num_random_sets, alpha)
    os.makedirs(os.path.dirname(table_file), exist_ok=True)

    W = net_obj.W
    M_inv = get_diffusion_matrix(W, alpha=alpha, diff_mat_file= "diffusion-alpha-%.2g.mat.npy" % alpha)

    nodes_idx = list(range(len(net_obj.nodes)))

    interactor_diff_values = {} 
    rand_diff_values = {}

    # Get the diffusion values for each virus interactor node and a set of random nodes
    # from all N - 1 remaining virus interactor nodes

    for cv_node_idx in interactor_nodes_idx:

        # Remove current interactor node as in leave one out cross validation
        cv_interactor_nodes_idx = list(interactor_nodes_idx)
        cv_interactor_nodes_idx.remove(cv_node_idx)
        
        cv_diff_value = sum(M_inv[cv_node_idx][cv_interactor_nodes_idx])
        interactor_diff_values[cv_node_idx] = cv_diff_value

        random_set = np.random.choice(nodes_idx, size=num_random_sets)
        rand_diff_values[cv_node_idx] = []
        for rand_idx in range(len(random_set)):
            rand_diff_value = sum(M_inv[rand_idx][cv_interactor_nodes_idx])
            rand_diff_values[cv_node_idx].append(rand_diff_value)

    print("writing %s" % (table_file))    
    with open(table_file, 'w') as table: 
        table.write('\t'.join(['UniProt ID', 'Gene ID', 'RL diffusion value', 'Random proteins mean RL diffusion value', 'p-value']) + '\n')
        for interactor_node_idx in interactor_diff_values:
            rank = 1
            for rand_node_idx in range(len(rand_diff_values[interactor_node_idx])): 
                if float(rand_diff_values[interactor_node_idx][rand_node_idx]) > float(interactor_diff_values[interactor_node_idx]):
                    rank += 1
            table.write('\t'.join([net_obj.nodes[interactor_node_idx], uniprot_to_gene[net_obj.nodes[interactor_node_idx]],
                '%.2g' % interactor_diff_values[interactor_node_idx],
                '%.2g' % (sum(rand_diff_values[interactor_node_idx]) / len(rand_diff_values[interactor_node_idx])), 
                       '%.2g' % (rank / len(rand_diff_values[interactor_node_idx]))]) + '\n') 
     
    interactor_values = pd.Series(list(interactor_diff_values.values()))
    rand_values = pd.Series(np.ravel(list(rand_diff_values.values())))

    return interactor_values, rand_values


def plot_diffusion_cross_validation(interactors, interact_values, rand_values, alpha, **kwargs):

    out_dir = kwargs.get('out_dir', 'rand_diffusion_cross_validation')
    plot_file = out_dir + "/%s-alpha-%.2g-rand-diffusion-cross-validation.pdf" % ("-".join(interactors), alpha)
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)

    for i in range(len(interactors)):
        statistic, pval = scipy.stats.ks_2samp(interact_values[i], rand_values[i], alternative='less')
        print("Kolmogorov-Smirnov p-value for diffusion scores between %s nodes and random nodes with alpha = %.2g: %.2g" % 
            (interactors[i], alpha, pval))

    plt.style.use('bmh')
    plt.figure(figsize = [12,2.5])
    maroon = ListedColormap(["maroon"])
    orange = ListedColormap(["darkorange"])
    for i in range(len(interactors)):
        ax = plt.subplot(1, len(interactors), i+1)

        interact_df = pd.DataFrame(columns = [interactors[i] + " Interactors"])
        interact_df[interactors[i] + " Interactors"] = interact_values[i]

        rand_df = pd.DataFrame(columns = ["Random Proteins"])
        rand_df["Random Proteins"] = rand_values[i]

        min_val = min([interact_values[i].min(), rand_values[i].min()])
        max_val = max([np.percentile(interact_values[i],98), np.percentile(interact_values[i],98)])
  
        interact_df.plot.kde(ax = ax, linewidth = 2, colormap = maroon)
        rand_df.plot.kde(ax = ax, linewidth = 2, style = ':', colormap = orange)
 
        ax.legend(loc='upper right')
        ax.tick_params(reset = 'True', axis = 'both', which = 'both', direction = 'out', length = 3, width = 1, right = False, top = False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlim(min_val, max_val)
        ax.set_ylabel("Probability Density")
        ax.set_xlabel("RL Scores")

    plt.tight_layout()
    print("writing %s" % (plot_file))
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    return

def get_diffusion_matrix(W, alpha=1.0, diff_mat_file=None):
    """
    Generate the diffusion matrix of a scipy sparse matrix
    *Note that the result is a dense matrix*
    """
    if diff_mat_file is not None and os.path.isfile(diff_mat_file):
        # read in the diffusion mat file
        print("Reading %s" % (diff_mat_file))
        return np.load(diff_mat_file)

    # now get the laplacian
    L = gm.setup_laplacian(W)
    # the equation is (I + a*L)s = y
    # we want to solve for (I + a*L)^-1
    M = sp.eye(L.shape[0]) + alpha*L 
    print("computing the inverse of (I + a*L) as the diffusion matrix, for alpha=%s" % (alpha))
    # first convert the scipy sparse matrix to a numpy matrix
    M_full = M.A
    # now try and take the inverse
    M_inv = scipy.linalg.inv(M_full)

    # write to file so this doesn't have to be recomputed
    if diff_mat_file is not None:
        print("Writing to %s" % (diff_mat_file))
        np.save(diff_mat_file, M_inv)

    return M_inv

if __name__ == "__main__":
    config_map_list, kwargs = parse_args()
    main(config_map_list, **kwargs)
