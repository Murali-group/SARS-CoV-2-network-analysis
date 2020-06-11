"""
Script to test for enrichment of any given list of genes (UniProt IDs)
"""

import argparse
import yaml
from collections import defaultdict
import os
import sys
import copy
import time
import pandas as pd


print("importing R packages")
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
utils_package = importr("utils")
clusterProfiler = importr('clusterProfiler')
ReactomePA = importr('ReactomePA')
base = importr('base')
base.require('org.Hs.eg.db')

import enrichment
import fss_enrichment

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)

    return kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to test for enrichment of the given gene/protein list among given genesets. " + \
                                     "Currently only tests for GO term enrichment")

    # general parameters
    group = parser.add_argument_group('Main Options')

    group.add_argument('--id-mapping-file', type=str, default="outputs/clustering/uniprot_gene_mapping.csv",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    group.add_argument('--prot-list-dir',default = 'outputs/clustering',
                       help="Test for enrichment of a list of proteins (UniProt IDs) (e.g., Krogan nodes).")
    group.add_argument('--prot-universe-file',
                       help="Protein universe to use when testing for enrichment")
    group.add_argument('--out-pref', default ="outputs/enrichment/clustering",
                       help="Output prefix where output files will be placed. " +
                       "Default is outputs/enrichement/<name_of_prots_list_file>")
    group.add_argument('--pval-cutoff', type=float, default=0.01,
                       help="Cutoff on the corrected p-value for enrichment.")
    group.add_argument('--qval-cutoff', type=float, default=0.05,
                       help="Cutoff on the Benjamini-Hochberg q-value for enrichment.")
    group.add_argument('--add-prot-list-to-prot-universe', action='store_true',
                       help="Add the positives listed in the pos_neg_file (e.g., Krogan nodes) to the prot universe ")
    group.add_argument('--force-run', action='store_true', default=False,
                       help="Force re-running the enrichment tests, and re-writing the output files")

    return parser


def main( **kwargs):
    """

    *kwargs*: all of the options passed into the script
    """

    # load the namespace mappings
    uniprot_to_gene = None
    gene_to_uniprot = None
    if kwargs.get('id_mapping_file'):
        # if kwargs.get('id_mapping_file'):
        uniprot_to_gene = uniprot_to_gene_from_one_to_one_map(kwargs.get('id_mapping_file'))
        kwargs['uniprot_to_gene'] = uniprot_to_gene


        gene_to_uniprot = gene_to_uniprot_from_one_to_one_map(kwargs.get('id_mapping_file'))
        kwargs['gene_to_uniprot'] = gene_to_uniprot

        # could not decide which id_mapping_file to use. using the Jeffe defined one to one mapping seemed good.
        # but for those I might not get a gene to uniprot mapping when needed in reactome.
        # so two mapping file can be needed. To make it less complicated, currently commenn the reactome analysis.
        # gene_to_uniprot = enrichment.load_uniprot(kwargs.get('id_mapping_file'))
        # kwargs['gene_to_uniprot'] = gene_to_uniprot


    prot_universe = None
    # load the protein universe file
    if kwargs.get('prot_universe_file') is not None:
        df = pd.read_csv(kwargs['prot_universe_file'], sep='\t', header=None)
        prot_universe = df[df.columns[0]]
        print("\t%d prots in universe" % (len(prot_universe)))


    out_pref = kwargs.get('out_pref')

    all_dfs = {g: pd.DataFrame() for g in ['BP', 'CC', 'MF']}
    all_dfs_KEGG = pd.DataFrame()
    all_dfs_reactome = pd.DataFrame()


    terms_to_keep_GO = {g: [] for g in ['BP', 'CC', 'MF']}
    pathways_to_keep_KEGG=[]
    pathways_to_keep_Reactome = []

    os.makedirs(os.path.dirname(out_pref), exist_ok=True)
    coCluster_input_dir = kwargs.get('prot_list_dir')

    for file in os.listdir(coCluster_input_dir):
        if 'genes.txt' in file:
            file_full_path = coCluster_input_dir+'/'+file
            df = pd.read_csv(file_full_path,header=None)

            prot = df.columns[0]
            prots_to_test = df[prot].apply(lambda x: gene_to_uniprot[x] )

            # print(prots_to_test)

            if kwargs.get('add_prot_list_to_prot_universe'):
                # make sure the list of proteins passed in are in the universe
                size_prot_universe = len(prot_universe)
                prot_universe = set(prots_to_test) | set(prot_universe)
                if len(prot_universe) != size_prot_universe:
                    print("\t%d prots from the prots_to_test added to the universe" % (len(prot_universe) - size_prot_universe))

            coCluster_ID = file.replace('-genes.txt','')
            print(coCluster_ID)
            out_dir = out_pref + '/' + coCluster_ID

            bp_df, mf_df, cc_df = enrichment.run_clusterProfiler_GO(
                prots_to_test, out_dir, prot_universe=prot_universe, forced=kwargs.get('force_run'), **kwargs)

            for ont, df in [('BP', bp_df), ('MF', mf_df), ('CC', cc_df)]:

                if not df.empty:
                    terms_to_keep_GO[ont] = terms_to_keep_GO[ont] + list(df[df['p.adjust']<=0.01]['ID'])
                tuples = [(coCluster_ID, '-', col) for col in df.columns]
                index = pd.MultiIndex.from_tuples(tuples)
                df.columns = index
                all_dfs[ont] = pd.concat([all_dfs[ont], df], axis=1)
                print(ont, len(all_dfs[ont]))
                # print('terms to keep', ont, terms_to_keep_GO[ont])


            # KEGG_df = enrichment.run_clusterProfiler_KEGG(prots_to_test, out_dir, prot_universe=prot_universe, forced=kwargs.get('force_run'),**kwargs)
            # if not KEGG_df.empty:
            #     pathways_to_keep_KEGG = pathways_to_keep_KEGG + list(KEGG_df[KEGG_df['p.adjust']<=0.01])
            # tuples = [(coCluster_ID, '-', col) for col in KEGG_df.columns]
            # index = pd.MultiIndex.from_tuples(tuples)
            # KEGG_df.columns = index
            # all_dfs_KEGG = pd.concat([all_dfs_KEGG, KEGG_df], axis=1)



            # reactome_df = enrichment.run_ReactomePA_Reactome(prots_to_test, out_dir, prot_universe=prot_universe, forced=kwargs.get('force_run'),**kwargs)
            # if not reactome_df.empty:
            #     pathways_to_keep_Reactome = pathways_to_keep_Reactome + list(reactome_df[reactome_df['p.adjust']<=0.01])
            # tuples = [(coCluster_ID, '-', col) for col in reactome_df.columns]
            # index = pd.MultiIndex.from_tuples(tuples)
            # reactome_df.columns = index
            # all_dfs_reactome = pd.concat([all_dfs_reactome, reactome_df], axis=1)
            #

    for geneset, g_df in all_dfs.items():

        all_dfs[geneset] = g_df[g_df.index.isin(terms_to_keep_GO[geneset])]

    # all_dfs_KEGG = all_dfs_KEGG[all_dfs_KEGG.index.isin(pathways_to_keep_KEGG)]
    #
    # all_dfs_reactome = all_dfs_reactome[all_dfs_reactome.index.isin(pathways_to_keep_Reactome)]


    # out_file = "%s/enrich-KEGG.csv" % (out_pref)
    # processed_df = fss_enrichment.write_combined_table(all_dfs_KEGG, out_file, dataset_level=0)
    #
    #
    # out_file = "%s/enrich-Reactome.csv" % (out_pref)
    # processed_df = fss_enrichment.write_combined_table(all_dfs_reactome, out_file, dataset_level=0)

    for geneset, df in all_dfs.items():
        out_file = "%s/enrich-GO-%s.csv" % (out_pref,geneset)
        processed_df= fss_enrichment.write_combined_table(df, out_file,dataset_level=0)


def gene_to_uniprot_from_one_to_one_map(id_mapping_file):
    """
    parameters: id_mapping_file: a file containing Uniprot_ID and geneName mapping, Jeff created it.
    returns: a dictionary where genename is the key and uniprot_id is the value
    """
    df = pd.read_csv(id_mapping_file, sep=',')
    ## keep only the first gene for each UniProt ID
    gene_to_uniprot = {gene: p for p, gene in zip(df['UniProt ID'], df['GeneName'].astype(str))}

    return gene_to_uniprot
def uniprot_to_gene_from_one_to_one_map(id_mapping_file):
    """
    parameters: id_mapping_file: a file containing Uniprot_ID and geneName mapping, Jeff created it.
    returns: a dictionary where genename is the key and uniprot_id is the value
    """
    df = pd.read_csv(id_mapping_file, sep=',')
    ## keep only the first gene for each UniProt ID
    uniprot_to_gene = {p: gene for p, gene in zip(df['UniProt ID'], df['GeneName'].astype(str))}

    return uniprot_to_gene


if __name__ == "__main__":
    kwargs = parse_args()
    main(**kwargs)
