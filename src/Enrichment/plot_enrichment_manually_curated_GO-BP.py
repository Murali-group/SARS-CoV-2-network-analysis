
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

    group.add_argument('--enrichmentdir',default = 'outputs/enrichment/combined-krogan-1_0/GO_BP')

    # group.add_argument('--order',type = list, default = ['RL', 'SVM', 'Kr'])
    group.add_argument('--compareRL', action='store_true', default = False )

    group.add_argument('--cluster', action='store_true', default = False )

    return parser



def plot_heatmap(df,algo_order, out_file_path):

    pval_cutoff  =0.01

    df.index.rename('',inplace = True)
    pval_cols = [col for col in df.columns if 'p.adjust' in col]

    # keep only the columns having '_pvalue' string in it
    df = df[pval_cols]
    print(df)

    for pval_col in pval_cols:
        df[pval_col] = df[pval_col].astype(float).apply(lambda x: x if x<pval_cutoff else np.nan )
        df[pval_col] = -np.log10(df[pval_col])
    # print(df)

    # change column names into GM+, SVM, Krogan from GM+_pvalue, SVM-rep100-nf5_pvalue,Krogan_pvalue
    changed_col_name = []

    for pval_col in pval_cols:
        changed_name = pval_col.replace('_p.adjust','').split('-')[0]
        if(changed_name == 'Krogan'):
            changed_name = 'Kr'
        if('CoClusters' in changed_name ):
            changed_name = changed_name.replace('CoClusters','bic')
        changed_col_name.append(changed_name)


    dict_for_rename = dict(zip(pval_cols, changed_col_name))
    df.rename(dict_for_rename, axis  = 1, inplace = True)

    df = df[algo_order]
    print(df.columns)

    fig_height = len(df.index)*(10/35)
    plt.figure(figsize=(5,fig_height))
    sns_plot = sns.heatmap(df, cmap='Blues',vmin =-np.log10(pval_cutoff),vmax = 20, cbar_kws={'label': '-log(pval)'}, square=True,linewidth=0.005)
    sns_plot.set_facecolor('xkcd:grey')

    out_png = out_file_path+'_heatmap.png'
    out_pdf = out_file_path+'_heatmap.pdf'


    plt.savefig(out_png, bbox_inches='tight', format = 'png')
    plt.savefig(out_pdf, bbox_inches='tight', format = 'pdf')

    plt.close()

def multi_level_to_single_level_enrichment_df(df):

    description = df['Description']['Unnamed: 1_level_1']['Unnamed: 1_level_2']
    df.drop('Description', level = 0, axis = 1, inplace = True)
    parsed_df = pd.DataFrame({'Description':description})

    for dataset, df_d in df.groupby(level = 0, axis = 1):

        for alg, df_a in df_d.groupby(level=1, axis = 1):

            df_a.columns = df_a.columns.droplevel([0,1])

            df_a['p.adjust']=df_a['p.adjust'].fillna(1)

            if(alg !='-'):

                adjust_pval_col = alg+'_'+'p.adjust'

            else:
                adjust_pval_col = dataset+'_'+'p.adjust'

            parsed_df[adjust_pval_col] = df_a['p.adjust']


    return  parsed_df

def filter_reindex(df, df1):

        df = multi_level_to_single_level_enrichment_df(df)

        df = df[df.index.isin(df1.index)]
        print('df shape before reindex: ', df.shape)

        # before reindexing make sure that df and df1 contains the same set of indices
        df1 = df1[df1.index.isin(df.index)]

        df = df.reindex(df1.index)
        print('df shape after reindex: ', df.shape)

        name_map= dict(zip(df['Description'], df1['Description']))

        df = df.set_index('Description')
        df.rename(index = name_map,inplace=True)

        return df

def main(**kwargs):
    enrichment_dir = kwargs.get('enrichmentdir')
    algo_order = ['RL']
    compare_RL= kwargs.get('compareRL')
    cluster= kwargs.get('cluster')
    print(enrichment_dir)

    file = 'string-k332-BP.csv'
    filter_file = 'string_k332_GO-BP_manually_simplified.csv'

    file_path  = enrichment_dir+'/'+file
    filter_file_path =  enrichment_dir+'/'+filter_file
    # extract GO,KEGG,Reactome from filename
    file_name_without_extension = os.path.splitext(file)[0]
    # enrichment_type = file.split('.|_')[-3]

    out_file = enrichment_dir + '/' + file_name_without_extension

    df1 = pd.read_csv(filter_file_path,index_col = 0)

    df = pd.read_csv(file_path, header = [0,1,2],index_col = 0)

    df = filter_reindex(df,df1.copy())

    # df = multi_level_to_single_level_enrichment_df(df)
    #
    # df = df[df.index.isin(df1.index)]
    #
    # df = df.reindex(df1.index)
    #
    # name_map= dict(zip(df['Description'], df1['Description']))
    #
    # df = df.set_index('Description')
    # df.rename(index = name_map,inplace=True)


    print(df)

    if(compare_RL):
        k_to_compare = [600]
        for k in k_to_compare:
            compare_file_path =enrichment_dir+'/'+'string-k'+str(k)+'-BP.csv'
            df_c =  pd.read_csv(compare_file_path, header = [0,1,2],index_col = 0)
            # df_c = multi_level_to_single_level_enrichment_df(df_c)
            #
            # df_c = df_c[df_c.index.isin(df1.index)]
            #
            # df_c = df_c.reindex(df1.index)
            #
            # name_map= dict(zip(df_c['Description'], df1['Description']))
            #
            # df_c = df_c.set_index('Description')
            #
            # df_c.rename(index = name_map,inplace=True)

            df_c = filter_reindex(df_c,df1.copy())

            RL_cols = [col for col in df_c.columns if 'RL' in col]
            df_c = df_c[RL_cols]

            # change column name prefix from RL to RL(value of k) : e.g. RL600
            changed_name_list = []
            RL_cols = df_c.columns
            for col in RL_cols:
                changed_name = col.replace('RL','RL'+str(k))
                changed_name_list.append(changed_name)
            df_c.rename(dict(zip(RL_cols,changed_name_list)), axis  = 1, inplace = True)


            df = pd.concat([df,df_c], axis = 1)
            algo_order.append('RL'+str(k))

    algo_order.append('SVM')
    algo_order.append('Kr')

    if cluster:
        cluster_enrichment_file_path = 'outputs/enrichment/clustering/enrich-GO-BP.csv'
        out_file =  'outputs/enrichment/clustering/string-k332-RL-SVM-Krogan-Cocluster'
        df_cluster = pd.read_csv(cluster_enrichment_file_path,header = [0,1,2],index_col = 0)
        df_cluster = filter_reindex(df_cluster,df1.copy())

        print('df_cluster: ',df_cluster)

        # print('df  shape: ' , df.shape)
        # print('df_cluster shape: ', df_cluster.shape)
        df = pd.concat([df,df_cluster], axis = 1)
        algo_order = algo_order + ['bic_1', 'bic_2', 'bic_3', 'bic_4', 'bic_5', 'bic_6', 'bic_7','bic_8']



    df.to_csv(enrichment_dir+'/'+'filtered_GO-BP.csv')

    plot_heatmap(df,algo_order, out_file)



if __name__ == "__main__":
    kwargs = parse_args()
    main(**kwargs)
