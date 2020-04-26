import scipy.stats as stats
import pandas as pd
import argparse
import yaml
import sys
import os
from src.setup_datasets import parse_gmt_file
sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        config_map = yaml.load(conf, Loader=yaml.FullLoader)
    return config_map, kwargs


def setup_opts():
    # Parse command line args.
    parser = argparse.ArgumentParser(description="Script for Fisher's exact test")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="config.yaml",
                       help="Configuration file for this script.")
    # NURE: allow multiple values to this argument since we want to trace the p-value as we increase k.
    # We can discuss on Slack. The code to plot the p-values as in my 2011 paper on HIV dependency factors can also be part of this script.
    # group.add_argument('--k', type=int, help="top k predictions to consider")

    group.add_argument('--ks', type=list, default=[100, 2000, 100], help="value of k will be from ks[0] to ks[1] increasing by amount of ks[2]")

    return parser


def parse_geneset(config_map):
    # this will filter out the positive Krogan gene sets
    # return* a mapping of gene_set_name to uniprotids of genes belonging to the protein set

    uniprotids, prot_descriptions = parse_gmt_file(config_map['reference_gene_set']['uniprotids'])
    for key in prot_descriptions:
        if "Krogan" in key:
            uniprotids.pop(key)
    return uniprotids



def calc_Fisher_exact_test_score(geneset_protein_list, predicted_protein_list, k):

    # Calculate Fisher's exact coefficient
    # returns:
    #       1. oddratio
    #       2. pvalue
    #       3. 4 values from contingency table

    total_number_of_protein = len(predicted_protein_list)

    # get the list of top-k predictions. Check if there are at least k predictions.
    if k < len(predicted_protein_list):
        predicted_top_k_proteins = predicted_protein_list[:k]
    else:
        predicted_top_k_proteins = predicted_protein_list
        k = len(predicted_top_k_proteins)

    # proteins that are present in the geneset but not in the entire network will be filtered out from the geneset here.
    reduced_geneset_protein_list = list(set(geneset_protein_list).intersection(set(predicted_protein_list)))

    proteins_common_in_reduced_geneset_and_top_k_prediction = list(set(reduced_geneset_protein_list).intersection(set(predicted_top_k_proteins)))

    a = len(proteins_common_in_reduced_geneset_and_top_k_prediction)
    b = len(reduced_geneset_protein_list) - a
    c = len(predicted_top_k_proteins) - a
    d = total_number_of_protein - a - b - c

    oddsratio, pvalue = stats.fisher_exact([[a, b], [c, d]],'greater')

    return oddsratio, pvalue,a, b, c, d


def get_list_of_positive_proteins(config_map):
    # returns: the list of Krogan proteins that have been used as positive examples in prediction
    positive_proteins = pd.read_csv(config_map['positive_protein_file'], '\t')
    return list(positive_proteins['prots'])

def combine_enrichment_scores(predicted_prot_dir):

    df = pd.DataFrame()
    for dirpath, dirs, files in os.walk(predicted_prot_dir):
        for filename in files:
            fname = os.path.join(dirpath, filename)

            if 'Fishers_exact' in fname:
                df_ = pd.read_csv(fname,'\t')
                df = df.append(df_)

    df.to_csv(predicted_prot_dir + '/combined_Fishers_test_score.csv','\t', index= False)


def main(config_map, **kwargs):

    # k = top k predictions to consider to find overlap
    k_list = []

    if (kwargs.get('ks')):
        ks = kwargs.get('ks')
        for k in range(ks[0], ks[1] + 1, ks[2]):
            k_list.append(k)

    geneset_uniprotids_dict = parse_geneset(config_map)

    # this is the Krogan protein list
    positive_protein_list = get_list_of_positive_proteins(config_map)

    predicted_prot_dir = config_map['predicted_prot_dir']

    for dirpath, dirs, files in os.walk(predicted_prot_dir):

        for filename in files:

            fname = os.path.join(dirpath, filename)

            # this code assume that pred-scores contain score for all the proteins in the network.
            if 'pred-scores' in fname:
                # if os.path.exists( dirpath + '/' + os.path.basename(dirpath) + '_Fishers_exact_test_score.csv'):
                #     continue
                # filtering out the positive proteins from predicted list of proteins
                predicted_prot_info = pd.read_csv(fname, sep='\t')
                predicted_protein_info = predicted_prot_info[~predicted_prot_info['prot'].isin(positive_protein_list)]
                predicted_protein_info = predicted_protein_info.sort_values(by=['score'],ascending = False)
                # print(predicted_protein_info['score'])
                predicted_protein_list = predicted_protein_info['prot']


                Fishers_exact_test_output_file = dirpath + '/' + os.path.basename(dirpath)+'_Fishers_exact_test_score.csv'
                dirpath_split = dirpath.split('/')
                algorithm = dirpath_split[-1]
                network_name = dirpath_split[-4]+'_'+dirpath_split[-3] + '_'+dirpath_split[-2]

                oddratio_list = []
                pvalue_list = []
                a_list=[]
                b_list=[]
                c_list=[]
                d_list=[]

                reference_prot_set_list = []
                rank = []
                fraction_of_positive_in_top_k_predicton_list = []
                network_name_list=[]
                algorithm_list=[]

                for gene_set_name in geneset_uniprotids_dict:

                    for k in k_list:

                        oddratio, pvalue, a, b, c, d = \
                            calc_Fisher_exact_test_score(geneset_uniprotids_dict[gene_set_name], predicted_protein_list, k)

                        reference_prot_set_list.append(gene_set_name)
                        rank.append(k)
                        a_list.append(a)
                        b_list.append(b)
                        c_list.append(c)
                        d_list.append(d)
                        oddratio_list.append(oddratio)
                        pvalue_list.append(pvalue)
                        fraction_of_positive_in_top_k_predicton_list.append(a/(a+c))
                        network_name_list.append(network_name)
                        algorithm_list.append(algorithm)

                print('writing to' + Fishers_exact_test_output_file)
                scores = pd.DataFrame({
                                        'network_name':pd.Series(network_name_list),
                                        'algorithm': pd.Series(algorithm_list),
                                        'protein_set_name': pd.Series(reference_prot_set_list),
                                        'rank':pd.Series(rank),
                                        'a': pd.Series(a_list),
                                        'b': pd.Series(b_list),
                                        'c': pd.Series(c_list),
                                        'd': pd.Series(d_list),
                                        'pvalue': pd.Series(pvalue_list),
                                        'oddratio': pd.Series(oddratio_list),
                                        'fraction_of_positive_in_top_k_predicton': pd.Series(fraction_of_positive_in_top_k_predicton_list)})
                scores.to_csv(Fishers_exact_test_output_file, '\t', index = False)

    combine_enrichment_scores(predicted_prot_dir)

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
