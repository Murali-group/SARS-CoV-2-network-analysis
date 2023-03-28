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
os.environ['R_HOME'] = '/home/tasnina/anaconda3/envs/enrichment/lib/R'


print("importing R packages")
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
utils_package = importr("utils")
clusterProfiler = importr('clusterProfiler')
# ReactomePA = importr('ReactomePA')
base = importr('base')
base.require('org.Hs.eg.db')

def run_clusterProfiler_GO(
        prot_universe, prots_to_test, ont, enrichment_str, out_dir, forced=False,  **kwargs):
    """
    parameters: prots_to_test: the list of predicted proteins(UniprotID)
                out_dir: Directory to write the output i.e. enrichment results
                prot_universe: List of background proteins or protein universe(UniprotID)
                forced: True means overwrite already existing results,
                False means use the existing results
                **kwargs: The clusterProfiler result only includes geneID, so
                 for geneNames we are passing the 'uniprot_to_gene map' via kwargs.
                This uniprotID to geneName mapping has to created outside this function.
    *returns*: a list of DataFrames of the enrichement of BP, MF, and CC
    """

    os.makedirs(out_dir, exist_ok=True)
    # TODO make this a seting
    print("Running enrichGO from clusterProfiler")


    out_file = "%s/enrich-%s-%s-%s.csv" % (out_dir, ont, enrichment_str,
                                           str(kwargs.get('pval_cutoff')).replace('.','_'))
    if forced is False and os.path.isfile(out_file):
        print("\t%s already exists. Use --force-run to overwrite" % (out_file))
        df = pd.read_csv(out_file, sep=',', index_col=None)
    else:
        out_file1 = "%s/enrich-temp-%s-%s-%s.csv" % \
                    (out_dir, ont, enrichment_str, str(kwargs.get('pval_cutoff')).replace('.', '_'))
        ego = clusterProfiler.enrichGO(
            gene          = StrVector(list(prots_to_test)),
            universe      = StrVector(list(prot_universe)),
            keyType       = 'UNIPROT',
            OrgDb         = base.get('org.Hs.eg.db'),
            ont           = ont,
            pAdjustMethod = "BH",
            pvalueCutoff  = kwargs.get('pval_cutoff'),
            qvalueCutoff  = kwargs.get('qval_cutoff')
            )
        utils_package.write_table(ego,out_file1, sep=",")
        df = pd.read_csv(out_file1, index_col=0)
        # get geneNames from geneIDs
        if not df.empty:
            gene_map = kwargs['uniprot_to_gene']
            df['geneName'] = df['geneID'].apply(lambda x: '/'.join([gene_map.get(p, p)\
                                for p in x.split('/')]))
        print('writing to: ', out_file)
        #TODO add k_to_test in filename
        df.to_csv(out_file, sep=',', index=False)
        os.remove(out_file1)
    return df


def load_gene_names(id_mapping_file):
    """
    parameters: id_mapping_file: a file containing Uniprot_ID and geneName mapping
    returns: a dictionary where uniprot_id is the key and genename is the value
    """
    df = pd.read_csv(id_mapping_file, sep='\t', header=0)
    ## keep only the first gene for each UniProt ID
    uniprot_to_gene = {p: genes.split(' ')[0] for p, genes in zip(df['Entry'],
                        df['Gene Names'].astype(str))}
    # if 'Protein names' in df.columns:
    #     uniprot_to_prot_names = dict(zip(df['Entry'], df['Protein names'].astype(str)))
    #     #node_desc = {n: {'Protein names': uniprot_to_prot_names[n]} for n in uniprot_to_prot_names}
    return uniprot_to_gene

def load_uniprot(id_mapping_file):
    """
    parameters: id_mapping_file: a file containing Uniprot_ID and geneName mapping
    returns: a dictionary where genename is the key and uniprot_id is the value
    """
    df = pd.read_csv(id_mapping_file, sep='\t', header=0)
    ## keep only the first gene for each UniProt ID
    gene_to_uniprot = {genes.split(' ')[0]: p for p, genes in zip(df['Entry'],
                                            df['Gene Names'].astype(str))}

    return gene_to_uniprot

