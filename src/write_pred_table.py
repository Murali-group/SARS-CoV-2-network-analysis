
import argparse
import yaml
from collections import defaultdict
import os
import sys
#from tqdm import tqdm
import copy
import time
import itertools
import numpy as np
#from scipy import sparse
import pandas as pd
# add this file's directory to the path so these imports work from anywhere
sys.path.insert(0,os.path.dirname(__file__))
#from FastSinkSource.run_eval_algs import setup_runners
from FastSinkSource.src.plot import plot_utils
from FastSinkSource.src.utils import config_utils
from FastSinkSource.src.algorithms import runner
from setup_datasets import parse_mapping_file


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    config_maps = []
    for config in args.config:
        with open(config, 'r') as conf:
            #config_map = yaml.load(conf, Loader=yaml.FullLoader)
            config_map = yaml.load(conf)
            config_maps.append(config_map)

    return config_maps, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to pull together predictions from multiple datasets and/or algorithms, and sort them by med rank")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, action="append",
                       help="Configuration file(s) used when running FastSinkSource")
    group.add_argument('--alg', '-A', dest='algs', type=str, action="append",
                       help="Algorithms for which to get results. Must be in the config file. " +
                       "If not specified, will get the list of algs with 'should_run' set to True in the config file")
    group.add_argument('--sample-neg-examples-factor', type=float, 
            help="If specified, sample negative examples randomly without replacement from the protein universe equal to <sample_neg_examples_factor> * # positives")
    group.add_argument('--num-reps', type=int, 
            help="Number of times to repeat the CV process. Default=1")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry' and 'Gene names'")
    #group.add_argument('--drug-id-mapping-file', type=str, 
    #                   help="Table parsed from DrugBank xml with drug names and other info")
    #group.add_argument('--drug-target-file', type=str,  # default="datasets/drug-targets/drugbank-v5.1.6/prot-drug-itxs.tsv"
    #                   help="Drug-target edge list that will be used to get the drug nodes and their targets (second column, tab-del)")
    group.add_argument('--drug-nodes-only', action='store_true', 
                       help="Drug-target edge list that will be used to get the drug nodes and their targets (second column, tab-del)")
    group.add_argument('--prot-drug-targets', action='store_true', 
                       help="Add the drugs that target each protein to the table")
    group.add_argument('--drug-info-file', type=str,  # default="datasets/drug-targets/drugbank-v5.1.6/prot-drug-itxs.tsv"
                       help="Tab-delimited file with extra information about the drugs (e.g., toxicity)")
    group.add_argument('--drug-target-info-file', type=str,  # default="datasets/drug-targets/drugbank-v5.1.6/prot-drug-itxs.tsv"
                       help="Information about the drug's affects on the target")
    group.add_argument('--num-pred-to-write', '-W', type=int, default=100,
            help="Number of predictions to keep. " +
            "If 0, none will be written. If -1, all will be written. Default=100")
    group.add_argument('--factor-pred-to-write', '-N', type=float, 
            help="Keep the predictions <factor>*num_pos. " +
            "For example, if the factor is 2, a term with 5 annotations would get the nodes with the top 10 prediction scores written to file.")
    group.add_argument('--out-pref', type=str, default="outputs/stats/pred-table-",
                       help="Output prefix for writing the table of predictions. Default=outputs/stats/pred-table-")
    group.add_argument('--round', type=int, default=3,
                       help="Round to the given number of decimal places in the output file")
    #group.add_argument('--download-only', action='store_true', default=False,
    #                   help="Stop once files are downloaded and mapped to UniProt IDs.")
    group.add_argument('--stat-sig-cutoff', type=float, 
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                       "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--apply-cutoff', action='store_true',
                       help="Only keep the nodes with a pval < stat_sig_cutoff")
    group.add_argument('--go-pos-neg-table-file', type=str, 
                       help="pos-neg table file containing a matrix of GO annotations")
    group.add_argument('--term', '-T', type=str, action="append",
                       help="Include a column per term specified (e.g., GO:0005886)")

#    # additional parameters
#    group = parser.add_argument_group('Additional options')
#    group.add_argument('--forcealg', action="store_true", default=False,
#            help="Force re-running algorithms if the output files already exist")
#    group.add_argument('--forcenet', action="store_true", default=False,
#            help="Force re-building network matrix from scratch")
#    group.add_argument('--verbose', action="store_true", default=False,
#            help="Print additional info about running times and such")

    return parser


def main(config_maps, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    uniprot_to_gene = None
    # also add the protein name
    uniprot_to_prot_names = None
    if kwargs.get('id_mapping_file'):
        print("Reading %s" % (kwargs['id_mapping_file']))
        df = pd.read_csv(kwargs['id_mapping_file'], sep='\t', header=0) 
        ## keep only the first gene for each UniProt ID
        uniprot_to_gene = {p: genes.split(' ')[0] for p, genes in zip(df['Entry'], df['Gene names'].astype(str))}
        if 'Protein names' in df.columns:
            uniprot_to_prot_names = dict(zip(df['Entry'], df['Protein names'].astype(str)))
    # if kwargs.get('drug_id_mapping_file'):
    #     print("Reading %s" % (kwargs['drug_id_mapping_file']))
    #     df = pd.read_csv(kwargs['drug_id_mapping_file'], sep='\t', header=0) 
    #     ## keep only the first gene for each UniProt ID
    #     uniprot_to_gene.update({d: name for d, name in zip(df['drugbank_id'], df['name'].astype(str))})
    #     # now get extra drug info
    #     uniprot_to_prot_names.update({d: groups for d, groups in zip(df['drugbank_id'], df['groups'].astype(str))})
    #     #if 'Protein names' in df.columns:
    #     #    uniprot_to_prot_names = dict(zip(df['Entry'], df['Protein names'].astype(str)))
    if kwargs.get('drug_info_file') is not None:
        # get toxicity, and other info from this file
        print("getting drug info from %s" % (kwargs['drug_info_file']))
        drug_df = pd.read_csv(kwargs['drug_info_file'], sep='\t')
        drug_names = {d: name for d, name in zip(drug_df['drugbank_id'], drug_df['name'].astype(str))}
        uniprot_to_gene.update(drug_names)
        drug_df.drop('name', axis=1, inplace=True)
        print(drug_df.head())
        drug_df.set_index('drugbank_id', inplace=True)
        print("\tadding these columns to the table: %s" % (drug_df.columns))
    # if kwargs.get('drug_target_file') is not None:
    #     # load the drug nodes to evaluate them separately
    #     print("getting drug nodes from %s" % (kwargs['drug_target_file']))
    #     df = pd.read_csv(kwargs['drug_target_file'], sep='\t')
    #     df['DrugTargets'] = df['UniProt ID'].map(uniprot_to_gene).astype(str)
    #     drug_targets = df.groupby("Drug ID")['DrugTargets'].apply(','.join)
    #     drug_targets = pd.DataFrame(drug_targets)
    #     drug_targets['TargetCount'] = df.groupby("Drug ID")['DrugTargets'].count()
    #     print(drug_targets.head())
    #     kwargs['drug_nodes'] = set(list(drug_targets.index.values))
    #     print("\t%d drug nodes" % (len(kwargs['drug_nodes'])))
    if kwargs.get('drug_target_info_file') is not None:
        # load the drug nodes to evaluate them separately
        print("getting drug target info from %s" % (kwargs['drug_target_info_file']))
        df = pd.read_csv(kwargs['drug_target_info_file'], sep='\t')
        # limit to the targets and humans
        df = df[(df['category'] == 'target') & (df['organism'] == 'Humans')].astype(str)
        df.drop(['category', 'organism'], axis=1, inplace=True)
        #df.replace('unknown','', inplace=True)
        #df.replace('nan','', inplace=True)
        # replace the uniprot IDs with gene names
        drug_target_genes = df['uniprot_id'].map(uniprot_to_gene).astype(str)
        # also add the protein names
        if uniprot_to_prot_names is not None:
            drug_target_prot_names = df['uniprot_id'].map(uniprot_to_prot_names).astype(str)
            df.insert(1, 'ProtNames', drug_target_prot_names)
        df.insert(1, 'DrugTargets', drug_target_genes)
        #df.drop('uniprot_id', axis=1, inplace=True)
        print(df.head())
        # for some reason, apply doesn't work on the entire dataframe
        # so apply it to each column individually
        #drug_targets = df.groupby("drugbank_id").apply(','.join)
        #drug_targets = df.astype(str).groupby("drugbank_id").transform(','.join)
        drug_targets = pd.DataFrame()
        # if specified, store the drugs that target each protein
        if kwargs.get('prot_drug_targets'):
            target_count = df.groupby("drugbank_id")['DrugTargets'].count().astype(str).to_dict()
            for col in df.columns:
                if col == 'uniprot_id':
                    continue
                drug_targets[col] = df.groupby("uniprot_id")[col].apply('|'.join)
            print(drug_targets.columns)
            print(drug_targets.head())
            # the number of targets per drug
            drug_targets['TargetCount'] = drug_targets['drugbank_id'].apply(lambda x: '|'.join(target_count[d] for d in x.split('|')))
            if kwargs.get('drug_info_file'):
                drug_targets['DrugName'] = drug_targets['drugbank_id'].apply(lambda x: '|'.join(drug_names[d] for d in x.split('|')))
            # now drop a couple of the columns to make it more manageable
            #drug_targets.drop(["ProtNames", "DrugTargets", ], axis=1, inplace=True)
            drug_targets = drug_targets[['drugbank_id', 'DrugName', 'TargetCount']]
            drug_targets.columns = ['DrugTargets', 'DrugNames', 'TargetCounts']
        else:
            # otherwise, store the targets for each drug
            for col in df.columns[1:]:
                drug_targets[col] = df.groupby("drugbank_id")[col].apply(','.join)
            drug_targets.insert(1, 'TargetCount', df.groupby("drugbank_id")['DrugTargets'].count())
            print(drug_targets.columns)
        #print(drug_targets.head())
        kwargs['drug_nodes'] = set(list(drug_targets.index.values))
        print("\t%d drug nodes" % (len(kwargs['drug_nodes'])))

    if kwargs.get('go_pos_neg_table_file') and kwargs.get('term'):
        print("Reading table of positive and negative annotations from %s" % (kwargs['go_pos_neg_table_file']))
        term_df = pd.read_csv(kwargs['go_pos_neg_table_file'], sep='\t', index_col=0)
        # get just the GO terms specified
        term_df = term_df[kwargs['term']]
        term_df.replace(1,"x", inplace=True)
        term_df.replace(0,"", inplace=True)
        term_df.replace(-1,"", inplace=True)
        # also load the summary file with term names
        summary_file = "datasets/go/2020-05/pos-neg/isa-partof-pos-neg-50-summary-stats.tsv"
        df2 = pd.read_csv(summary_file, sep='\t', index_col=0)
        term_names = dict(zip(df2.index, df2['GO term name']))
        term_df.columns = ["%s (%s)" %(term_names[c], c) for c in term_df.columns]
        print(term_df.head())

    # for each dataset, get the prediction scores from each method
    # store a single df for each alg
    df_all = pd.DataFrame()
    all_datasets = []
    for config_map in config_maps:
        input_settings, input_dir, output_dir, alg_settings, kwargs \
            = config_utils.setup_config_variables(config_map, **kwargs)
        kwargs['algs'] = plot_utils.get_algs_to_run(alg_settings, **kwargs)

        all_datasets += input_settings['datasets']
        for dataset in input_settings['datasets']:
            # load the positive examples and remove them from the predictions
            pos_neg_file = "%s/%s" % (input_dir, dataset['pos_neg_file'])
            df = pd.read_csv(pos_neg_file, sep='\t')
            orig_pos = df[df['2020-03-sarscov2-human-ppi'] == 1]['prots']

            dataset_name = config_utils.get_dataset_name(dataset) 
            alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                output_dir, dataset, alg_settings, **kwargs)
            for alg, pred_file in alg_pred_files.items():
                if not os.path.isfile(pred_file):
                    print("Warning: %s not found. skipping" % (pred_file))
                    continue
                print("reading: %s" % (pred_file))
                df = pd.read_csv(pred_file, sep='\t')
                # remove the original positives
                df = df[~df['prot'].isin(orig_pos)]
                df.reset_index(inplace=True, drop=True)
                if kwargs.get('drug_nodes_only'):
                    df = df[df['prot'].isin(kwargs['drug_nodes'])]
                    print("\tlimited to %d drugs" % (len(df)))
                    df.reset_index(inplace=True, drop=True)

                if kwargs.get('stat_sig_cutoff'):
                    df = config_utils.get_pvals_apply_cutoff(df, pred_file, **kwargs)
                    # prot is the index, so move it to a column
                    df.reset_index(inplace=True)
                    df = df[['prot', 'pval']]
                    # reset the index again to store the current rank as a column
                    df.reset_index(inplace=True)
                    df.columns = ['Rank', 'Prot', 'Pval']
                else:
                    df = df[['prot', 'score']]
                    # reset the index again to store the current rank as a column
                    df.reset_index(inplace=True)
                    df.columns = ['Rank', 'Prot', 'Score']
                df['Rank'] += 1

                # now set the index as the uniprot ID 
                df.set_index('Prot', inplace=True)

                # add the dataset-specific settings to the column headers
                #df.columns = ["%s-%s" % (curr_name, col) for col in ["Rank", "Score"]]
                # UPDATE: add levels to the column index
                tuples = [(dataset_name, alg, col) for col in df.columns]
                index = pd.MultiIndex.from_tuples(tuples)
                df.columns = index
                print(df.head())

                # now add the columns to the master dataframe
                df_all = pd.concat([df_all, df], axis=1)
                #for col in df.columns:
                #    alg_dfs[alg][col] = df[col]

    print(df_all.head())

    # sort the dataframe by the average rank
    # first compute the ranks, then add them to the dataframe later
    print("Sorting genes by average rank")
    # for each dataset, replace NA with the highest rank in that network
    df_ranks = df_all[[col for col in df_all.columns if 'Rank' in col]]
    df_ranks.fillna(df_ranks.max(axis=0), inplace=True)
    # not exactly sure what the "bottom" option does from pandas
    #           .rank(method='average', na_option='bottom')

    #nan_rank_val = len(orig_pos)*2
    #print("\tempty ranks getting a value of %d" % (nan_rank_val))
    #df_ranks = df_all[[col for col in df_all.columns if 'Rank' in col]].fillna(nan_rank_val)
    #print(df_ranks.head())
    med_rank = df_ranks.median(axis=1).sort_values()
    avg_rank = df_ranks.mean(axis=1).sort_values()
    med_rank = med_rank.round(1)
    avg_rank = avg_rank.round(1)
    #print(med_rank)
    #print(len(med_rank))

    # figure out how many predictions to keep
    num_pred_to_write = kwargs.get('num_pred_to_write', 100) 
    if kwargs.get('factor_pred_to_write') is not None:
        num_pred_to_write = kwargs['factor_pred_to_write']*len(orig_pos)

    if num_pred_to_write == -1:
        print("Keeping all predictions from each method")
    else:
        print('Keeping %d predictions from each alg' % (num_pred_to_write))
        # for each dataset, keep the top k predictions, with ties
        for dataset in all_datasets:
            name = dataset['plot_exp_name']
            curr_algs = df_all[name].columns.levels[0]
            for alg in curr_algs:
                sub_df = df_all[(name, alg)]
                sub_df.sort_values(by='Rank', inplace=True)

                topk = int(min(len(sub_df), num_pred_to_write))
                # instead of the score, just use the rank, since there are most likely no ties, and the score column could be replaced by the pval column
                #score_topk = sub_df.iloc[topk-1]['Score']
                #sub_df = sub_df[sub_df['Score'] >= score_topk]
                sub_df = sub_df.iloc[:topk]
                # TODO for some reason this doesn't work if that are nan or inf in the column.
                #sub_df['Score'] = sub_df['Score'].round(kwargs.get('round', 3))
                # use object to keep the value as an integer in the output
                sub_df['Rank'] = sub_df['Rank'].astype(int).astype(object)
                df_all[[(name, alg, col) for col in sub_df.columns]] = sub_df

    # now remove rows that are NA for all values
    df_all.dropna(how='all', inplace=True)

    if kwargs.get('go_pos_neg_table_file') and kwargs.get('term'):
        # Add the extra term columns
        for col in term_df.columns[::-1]:
            df_all.insert(0, col, term_df[col])

    if kwargs.get('drug_info_file') and not kwargs.get('prot_drug_targets'):
        # Add the extra drug columns
        for col in drug_df.columns[::-1]:
            df_all.insert(0, col, drug_df[col])
    if kwargs.get('drug_target_info_file'):
        for col in drug_targets.columns[::-1]:
            df_all.insert(0, col, drug_targets[col])

    # and the average rank
    df_all.insert(0, 'AverageRank', avg_rank)
    df_all.insert(0, 'MedianRank', med_rank)
    df_all.sort_values(by='MedianRank', inplace=True)
    # add the protein names
    if uniprot_to_prot_names is not None:
        df_all.insert(0, 'ProteinNames', df_all.index.map(uniprot_to_prot_names))
    # add the gene names
    if uniprot_to_gene is not None:
        df_all.insert(0, 'GeneName', df_all.index.map(uniprot_to_gene))

    print(df_all.head())

    # make sure the output directory exists
    os.makedirs(os.path.dirname(kwargs['out_pref']), exist_ok=True)
    #for alg, df in alg_dfs.items():
    out_file = "%s%s.tsv" % (kwargs['out_pref'], '-'.join(kwargs['algs']))
    print("\nWriting %s" % (out_file))
    df_all.to_csv(out_file, sep='\t')


if __name__ == "__main__":
    config_maps, kwargs = parse_args()
    main(config_maps, **kwargs)
