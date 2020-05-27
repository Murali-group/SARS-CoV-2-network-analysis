
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

    group.add_argument('--enrichmentdir',default = 'outputs/enrichment/combined-krogan-1_0/simplified')

    group.add_argument('--order',type = list, default = ['GM+', 'SVM', 'Krogan'])

    return parser



def plot_heatmap(df,algo_order, out_file_path):
    # df.reset_index(inplace = True)
    # df.set_index(['index','Description'],inplace=True)
    # df.drop('Description')
    # df.rename(columns={'Unnamed: 0': 'Term/pathway ID'})
    # df.set_index('Term/pathway ID', inplace =True)

    # set 'Description' column to be index
    # df.set_index('Description', inplace=True)

    print(df.columns)

    df.index.rename('',inplace = True)
    pval_cols = [col for col in df.columns if 'pvalue' in col]

    # keep only the columns having '_pvalue' string in it
    df = df[pval_cols]
    print(df)

    for pval_col in pval_cols:
        df[pval_col] = -np.log10(df[pval_col])

    # change column names into GM+, SVM, Krogan from GM+_pvalue, SVM-rep100-nf5_pvalue,Krogan_pvalue
    changed_col_name = []
    for pval_col in pval_cols:
        changed_name = pval_col.replace('_pvalue','').split('-')[0]
        changed_col_name.append(changed_name)

    dict_for_rename = dict(zip(pval_cols, changed_col_name))
    df.rename(dict_for_rename, axis  = 1, inplace = True)

    df = df[algo_order]


    print(df)
    sns_plot = sns.heatmap(df, cmap="Blues",cbar_kws={'label': '-log(pval)'})
    plt.savefig(out_file_path, bbox_inches='tight')
    plt.close()


def main(**kwargs):
    enrichment_dir = kwargs.get('enrichmentdir')
    algo_order = kwargs.get('order')
    print(enrichment_dir)


    for file in os.listdir(enrichment_dir):

        if '.csv' in file:
            file_path  = enrichment_dir+'/'+file
            # extract GO,KEGG,Reactome from filename
            # enrichment_type = file.split('.|_')[-3]
            file_name_without_extension = os.path.splitext(file)[0]
            # enrichment_type = file.split('.|_')[-3]

            out_file = enrichment_dir + '/' + file_name_without_extension + '_heatmap.png'

            # index_col = 1 which is the Description column
            df = pd.read_csv(file_path, index_col = 1)

            plot_heatmap(df,algo_order, out_file)



if __name__ == "__main__":
    kwargs = parse_args()
    main(**kwargs)
