
import argparse
import yaml
from collections import defaultdict
import os
import sys
#from tqdm import tqdm
import copy
import time
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
#from scipy import sparse
import pandas as pd
import numpy as np

sys.path.insert(1, '/home/tasnina/SARS-CoV-2-network-analysis')

from src.utils import parse_utils as utils
from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)

    return kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to test for enrichment of the top predictions among given genesets. " + \
                                     "Currently only tests for GO term enrichment")

    # general parameters
    group = parser.add_argument_group('Main Options')

    group.add_argument('--enrichmentfile',default = 'outputs/enrichment/combined-krogan-1_0/greedy_simplified/string_k332_GO-BP_simplified.csv')

    # group.add_argument('--order',type = list, default = ['RL-a0_01','RL-a0_1','RL-a0_5','RL-a1','RL-a2','RL-a10','RL-a100','SVM', 'Kr'])

    group.add_argument('--multAlpha',type = bool, default = False)


    return parser



def plot_heatmap_all_algo_krogan(df,algo_order, out_file_path,mult_alpha):

    pval_cutoff  =0.01

    print(df.columns)

    df.index.rename('',inplace = True)
    pval_cols = [col for col in df.columns if 'p.adjust' in col]

    # keep only the columns having '_pvalue' string in it
    df = df[pval_cols]
    print(df)

    for pval_col in pval_cols:
        df[pval_col] = df[pval_col].astype(float).apply(lambda x: x if x<pval_cutoff else np.nan )
        df[pval_col] = -np.log10(df[pval_col])
    # print(df)

    # change column names into RL, SVM, Krogan from RL_pvalue, SVM-rep100-nf5_pvalue,Krogan_pvalue
    changed_col_name = []

    for pval_col in pval_cols:
        changed_name = pval_col.replace('_p.adjust','').split('-')

        if changed_name[0]=='RL':
            if mult_alpha:
                changed_name = '-'.join(changed_name[:2])
            else:
                changed_name = 'RL'
        elif(changed_name[0] == 'Krogan'):
            changed_name = 'Kr'
        elif(changed_name[0] == 'SVM'):
            changed_name = 'SVM'
        changed_col_name.append(changed_name)

    dict_for_rename = dict(zip(pval_cols, changed_col_name))
    print('Dict for rename:', dict_for_rename)
    df.rename(dict_for_rename, axis  = 1, inplace = True)

    df = df[algo_order]

    fig_height = len(df.index)*(10/35)
    plt.figure(figsize=(5,fig_height))
    sns_plot = sns.heatmap(df, cmap='Blues',vmin =-np.log10(pval_cutoff),vmax = 20, cbar_kws={'label': '-log(pval)'}, square=True,linewidth=0.005)
    sns_plot.set_facecolor('xkcd:grey')

    out_png = out_file_path+'_heatmap.png'
    out_pdf = out_file_path+'_heatmap.pdf'

    plt.savefig(out_png, bbox_inches='tight', format = 'png')
    plt.savefig(out_pdf, bbox_inches='tight', format = 'pdf')

    plt.close()


def main(**kwargs):
    enrichment_file = kwargs.get('enrichmentfile')
    mult_alpha = kwargs.get('multAlpha')
    if mult_alpha:
        algo_order = ['RL-a0_01','RL-a0_1','RL-a0_5','RL-a1','RL-a2','RL-a10','RL-a100']
    else:
        algo_order = ['RL','SVM', 'Kr']

    # if enrichment_dir == 'outputs/enrichment/combined-krogan-1_0/greedy_simplified':
    enrichment_dir = os.path.dirname(enrichment_file)
    file_name_without_extension = os.path.splitext(enrichment_file)[0]
    print('filename without extension: ', file_name_without_extension)
    # enrichment_type = file.split('.|_')[-3]

    out_file = file_name_without_extension

    # index_col = 1 which is the Description column
    df = pd.read_csv(enrichment_file, index_col = 1)

    plot_heatmap_all_algo_krogan(df,algo_order, out_file,mult_alpha)


if __name__ == "__main__":
    kwargs = parse_args()
    main(**kwargs)
