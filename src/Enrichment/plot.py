import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
import pandas as pd
import sys
import os
from src.Enrichment.enrichment_analysis import parse_args, setup_opts
sys.path.insert(0, os.path.dirname(__file__))


def plot_per_network_per_algo_per_geneset(score_dirpath):

    for dirpath, dirs, files in os.walk(score_dirpath):

        for filename in files:
            fname = os.path.join(dirpath, filename)
            if 'Fishers_exact' in fname:
                if not os.path.exists(dirpath + '/Plots'):
                    os.mkdir(dirpath + '/Plots')
                if not os.path.exists(dirpath + '/Plots/pvalue'):
                    os.mkdir(dirpath + '/Plots/pvalue')
                if not os.path.exists(dirpath + '/Plots/fraction_of_positive_in_top_k_predicton'):
                    os.mkdir(dirpath + '/Plots/fraction_of_positive_in_top_k_predicton')

                print(fname)
                df = pd.read_csv(fname,'\t')
                protein_set_list = df.protein_set_name.unique()
                for protein_set_name in protein_set_list:
                    df_ = df.loc[df['protein_set_name'] == protein_set_name]
                    df_significant = df_.loc[df_['pvalue'] <= 0.05]
                    df_insignificant = df_.loc[df_['pvalue'] > 0.05]
                    plot_pvalue(df_significant, df_insignificant,protein_set_name, dirpath)
                    plot_fraction(df_significant, df_insignificant, protein_set_name, dirpath)
                print('done with' + fname)


def plot_pvalue(significant, insignificant, protein_set_name, saving_dirpath):

    min_rank = 100
    max_rank = 2000

    # print(significant['rank'].to_numpy())

    fig, ax = plt.subplots()
    ax.plot(significant['rank'].to_numpy(),significant['pvalue'].to_numpy(), '--o', insignificant['rank'].to_numpy(), insignificant['pvalue'].to_numpy(), '--o')

    # plt.plot(significant['rank'].to_numpy(),significant['pvalue'].to_numpy(), '--o')
    plt.xlabel('rank')
    plt.ylabel('p-value')
    plt.title("\n".join(wrap('overlap between ' + protein_set_name + ' and proteins predicted by ' + os.path.basename(saving_dirpath))))
    plt.xticks(np.arange(0, max_rank+1, 200))
    protein_set_name = protein_set_name.replace("/", "_")
    plt.savefig(saving_dirpath + "/Plots/" + 'pvalue/' + protein_set_name, format='png')
    plt.close()

def plot_fraction(significant, insignificant, protein_set_name, saving_dirpath):

    min_rank = 100
    max_rank = 2000

    # print(significant['rank'].to_numpy())

    fig, ax = plt.subplots()
    ax.plot(significant['rank'].to_numpy(),significant['fraction_of_positive_in_top_k_predicton'].to_numpy(), '--o',
            insignificant['rank'].to_numpy(), insignificant['fraction_of_positive_in_top_k_predicton'].to_numpy(), '--o')

    # plt.plot(significant['rank'].to_numpy(),significant['pvalue'].to_numpy(), '--o')
    plt.xlabel('rank')
    plt.ylabel('fraction_of_positive_in_top_k_predicton')
    plt.title("\n".join(wrap('overlap between ' + protein_set_name + ' and proteins predicted by ' + os.path.basename(saving_dirpath))))
    plt.xticks(np.arange(0, max_rank+1, 200))
    protein_set_name = protein_set_name.replace("/", "_")
    plt.savefig(saving_dirpath + "/Plots/" + 'fraction_of_positive_in_top_k_predicton/' + protein_set_name, format='png')
    plt.close()

def plot_per_network_per_algo_per_k(combined_test_score_directory,combined_test_score_file,k_list):

    if not os.path.exists(combined_test_score_directory + '/Plots'):
        os.mkdir(combined_test_score_directory + '/Plots')

    df = pd.read_csv(combined_test_score_directory+'/'+combined_test_score_file, '\t')

    network_list = df.network_name.unique()
    algo_list = df.algorithm.unique()

    for network_name in network_list:
        df_net = df.loc[df['network_name'] == network_name]
        for algo_name in algo_list:
            df_algo = df_net.loc[df_net['algorithm'] == algo_name]

            if not os.path.exists(combined_test_score_directory + '/Plots/'+network_name):
                os.mkdir(combined_test_score_directory + '/Plots/'+network_name)

            if not os.path.exists(combined_test_score_directory + '/Plots/'+network_name+'/'+algo_name):
                os.mkdir(combined_test_score_directory + '/Plots/'+network_name+'/'+algo_name)

            for k in k_list:
                df_k = df_algo.loc[df_algo['rank'] == k]
                print(len(df_k))

                plt.hist(df_k['pvalue'].to_numpy(),bins = 10)
                plt.xlabel('p-value')
                plt.ylabel('# of genesets')
                plt.title("\n".join(wrap(network_name +' + '+ algo_name+'+ k_'+ str(k))))

                plt.savefig(combined_test_score_directory + '/Plots/'+network_name+'/'+algo_name+'/' + network_name +'_' +algo_name +'_' + str(k),format='png')
                plt.close()
            print('done with ' + algo_name)


def plot_per_network(combined_test_score_directory,combined_test_score_file,k_list):

    if not os.path.exists(combined_test_score_directory + '/Plots'):
        os.mkdir(combined_test_score_directory + '/Plots')

    df = pd.read_csv(combined_test_score_directory + '/' + combined_test_score_file, '\t')
    geneset_list = df.protein_set_name.unique()

    df_significant = df.loc[df['pvalue']<=0.05]
    network_list = df_significant.network_name.unique()
    algo_list = df_significant.algorithm.unique()

    for network_name in network_list:
        if not os.path.exists(combined_test_score_directory + '/Plots/' + network_name):
            os.mkdir(combined_test_score_directory + '/Plots/' + network_name)

        df_net = df_significant.loc[df_significant['network_name'] == network_name]
        for algo_name in algo_list:
            df_algo = df_net.loc[df_net['algorithm'] == algo_name]
            y_value = []
            for k in k_list:
                y_val = len(df_algo.loc[df_algo['rank'] == k]) / len(geneset_list)
                y_value.append(y_val)
            p = plt.plot(k_list,y_value,'--o',label = algo_name)

        plt.xlabel('rank')
        plt.ylabel('fraction of geneset having significant p-value')
        plt.title("\n".join(wrap(network_name)))
        # for i in range(len(algo_list)):
        legend = plt.legend(loc='upper left', shadow=False, fontsize='xx-small',borderpad = 0.1)
        plt.savefig(combined_test_score_directory +  '/Plots/' + network_name+'/'+ network_name,
                    format='png')
        plt.close()


def main(config_map, **kwargs):

    # plot_per_network_per_algo_per_geneset(config_map['predicted_prot_dir'])

    k_list = []
    if kwargs.get('ks'):
        ks = kwargs.get('ks')
        for k in range(ks[0], ks[1] + 1, ks[2]):
            k_list.append(k)
    print(k_list)

    # plot_per_network_per_algo_per_k(config_map['combined_test_score_directory'],config_map['combined_test_score_file'], k_list)
    # plot_per_network(config_map['combined_test_score_directory'], config_map['combined_test_score_file'], k_list)

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map,**kwargs)
