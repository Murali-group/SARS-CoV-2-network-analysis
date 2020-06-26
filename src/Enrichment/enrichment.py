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

# # packages in this repo
# sys.path.insert(1, '/home/tasnina/SARS-CoV-2-network-analysis')
# # from src.utils import parse_utils as utils
# # from src.FastSinkSource.src import main as run_eval_algs
# # from src.FastSinkSource.src.utils import config_utils
# # from src.FastSinkSource.src.algorithms import alg_utils
# import david_client


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
    group.add_argument('--config', type=str, required=True,
                       help="Configuration file used when running FSS. " +
                       "Must have a 'genesets_to_test' section for this script. ")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    group.add_argument('--prot-list-file',
                       help="Test for enrichment of a list of proteins (UniProt IDs) (e.g., Krogan nodes).")
    group.add_argument('--prot-universe-file',
                       help="Protein universe to use when testing for enrichment")
    group.add_argument('--out-pref',
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
        uniprot_to_gene = load_gene_names(kwargs.get('id_mapping_file'))
        kwargs['uniprot_to_gene'] = uniprot_to_gene

        gene_to_uniprot = load_uniprot(kwargs.get('id_mapping_file'))
        kwargs['gene_to_uniprot'] = gene_to_uniprot

    df = pd.read_csv(kwargs['prot_list_file'], sep='\t', header=None)
    print(df)
    prots_to_test = list(df[df.columns[0]])
    # print("%d prots for which to test enrichment. (top 10: %s)" % (len(prots_to_test), prots_to_test[:10]))
    prot_universe = None
    # load the protein universe file
    if kwargs.get('prot_universe_file') is not None:
        df = pd.read_csv(kwargs['prot_universe_file'], sep='\t', header=None)
        prot_universe = df[df.columns[0]]
        # print("\t%d prots in universe" % (len(prot_universe)))
        if kwargs.get('add_prot_list_to_prot_universe'):
            # make sure the list of proteins passed in are in the universe
            size_prot_universe = len(prot_universe)
            prot_universe = set(prots_to_test) | set(prot_universe)
            if len(prot_universe) != size_prot_universe:
                print("\t%d prots from the prots_to_test added to the universe" % (len(prot_universe) - size_prot_universe))

    out_pref = kwargs.get('out_pref')
    if out_pref is None:
        out_pref = "outputs/enrichment/%s" % (os.path.basename(kwargs['prot_list_file']).split('.')[0])

    bp_df, mf_df, cc_df = run_clusterProfiler_GO(
        prots_to_test, out_pref, prot_universe=prot_universe, forced=kwargs.get('force_run'), **kwargs)

    KEGG_df = run_clusterProfiler_KEGG(prots_to_test, out_pref, prot_universe=prot_universe, forced=kwargs.get('force_run'),**kwargs)

    reactome_df = run_ReactomePA_Reactome(prots_to_test, out_pref, prot_universe=prot_universe, forced=kwargs.get('force_run'),**kwargs)



def run_clusterProfiler_GO(
        prots_to_test, out_dir, prot_universe=None, forced=False, **kwargs):
    """
    parameters: prots_to_test: the list of predicted proteins(UniprotID)
                out_dir: Directory to write the output i.e. enrichment results
                prot_universe: List of background proteins or protein universe(UniprotID)
                forced: True means overwrite already existing results, False means use the existing results
                **kwargs: The clusterProfiler result only includes geneID, so for geneNames we are passing the 'uniprot_to_gene map' via kwargs.
                          This uniprotID to geneName mapping has to created outside this function.
    *returns*: a list of DataFrames of the enrichement of BP, MF, and CC
    """
    os.makedirs(out_dir, exist_ok=True)
    # now load those results and make a table
    # TODO make this a seting
    print("Running enrichGO from clusterProfiler")
    if prot_universe is None:
        print("ERROR: default prot universe not yet implemented. Quitting")
        sys.exit()


    for ont in ['BP', 'MF', 'CC']:

        out_file = "%s/enrich-%s-%s.csv" % (out_dir,ont,str(kwargs.get('pval_cutoff')).replace('.','_'))
        if forced is False and os.path.isfile(out_file):
            print("\t%s already exists. Use --force-run to overwrite" % (out_file))
            continue

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
        utils_package.write_table(ego,out_file, sep=",")

    ont_dfs = []
    for ont in ['BP', 'MF', 'CC']:
        out_file = "%s/enrich-%s-%s.csv" % (out_dir,ont,str(kwargs.get('pval_cutoff')).replace('.','_'))
        # print("\treading %s" % (out_file))
        df = pd.read_csv(out_file, index_col=0)

        # get geneNames from geneIDs
        if not df.empty:
            gene_map = kwargs['uniprot_to_gene']
            df['geneName'] = df['geneID'].apply(lambda x: '/'.join([gene_map.get(p,p) for p in x.split('/')]))
        df.to_csv(out_file)
        ont_dfs.append(df)
    return ont_dfs

def run_clusterProfiler_KEGG(
        prots_to_test, out_dir, prot_universe=None, forced=False, **kwargs):
    """
    parameters: prots_to_test: the list of predicted proteins(UniprotID)
                out_dir: Directory to write the output i.e. enrichment results
                prot_universe: List of background proteins or protein universe(UniprotID)
                forced: True means overwrite already existing results, False means use the existing results
                **kwargs: The clusterProfiler result only includes geneID, so for geneNames we are passing the 'uniprot_to_gene map' via kwargs.
                          This uniprotID to geneName mapping has to created outside this function.
    *returns*: a DataFrames of the enrichement of KEGG pathways
    """
    os.makedirs(out_dir, exist_ok=True)
    #out_file = "%s/enrich-KEGG.csv" % (out_dir)

    # TODO make this a seting
    print("Running enrichKEGG from clusterProfiler")
    if prot_universe is None:
        print("ERROR: default prot universe not yet implemented. Quitting")
        sys.exit()

    out_file = "%s/enrich-KEGG-%s.csv" % (out_dir,str(kwargs.get('pval_cutoff')).replace('.','_'))
    if forced is False and os.path.isfile(out_file):
        print("\t%s already exists. Use --force-run to overwrite" % (out_file))
    else:
        e_kegg = clusterProfiler.enrichKEGG(
        gene = StrVector(list(prots_to_test)),
        universe = StrVector(list(prot_universe)),
        keyType = 'uniprot',
        organism="hsa",
        pAdjustMethod="BH",
        pvalueCutoff  = kwargs.get('pval_cutoff'),
        qvalueCutoff  = kwargs.get('qval_cutoff')
        )

        # print("\twriting %s" % (out_file))
        utils_package.write_csv(e_kegg,out_file)
        df = pd.read_csv(out_file, index_col=0)

        # get geneNames from geneIDs
        if not df.empty:
            gene_map = kwargs['uniprot_to_gene']
            df['geneName'] = df['geneID'].apply(lambda x: '/'.join([gene_map.get(p,p) for p in x.split('/')]))

        df.to_csv(out_file)

    df = pd.read_csv(out_file, index_col=0)

    return df

def run_ReactomePA_Reactome(
        prots_to_test, out_dir, prot_universe=None, forced=False,**kwargs):

    """
    parameters: prots_to_test: the list of predicted proteins(UniprotID)
                out_dir: Directory to write the output i.e. enrichment results
                prot_universe: List of background proteins or protein universe(UniprotID)
                forced: True means overwrite already existing results, False means use the existing results
                **kwargs: The ReactomePA package result only includes geneName, so for geneIDs we are passing the 'gene_to_uniprot' map via kwargs.
                          This geneName to uniprot mapping has to created outside this function.
    *returns*: a DataFrames of the enrichement of Reactome pathways
    """
    os.makedirs(out_dir, exist_ok=True)
    #out_file = "%s/enrich-KEGG.csv" % (out_dir)

    # TODO make this a seting
    print("Running enrich Reactome from ReactomePA")
    if prot_universe is None:
        print("ERROR: default prot universe not yet implemented. Quitting")
        sys.exit()

    out_file = "%s/enrich-Reactome-%s.csv" % (out_dir,str(kwargs.get('pval_cutoff')).replace('.','_'))

    if forced is False and os.path.isfile(out_file):
        print("\t%s already exists. Use --force-run to overwrite" % (out_file))
    else:
        # mapping from uniprot to EntrezID
        test_prot_uniprot_to_entrez_id_mapping = clusterProfiler.bitr(StrVector(list(prots_to_test)), fromType = 'UNIPROT', toType = "ENTREZID", OrgDb = base.get('org.Hs.eg.db'))
        test_prot_entrez_ids =  test_prot_uniprot_to_entrez_id_mapping[1]

        universe_prot_uniprot_to_entrez_id_mapping = clusterProfiler.bitr(StrVector(list(prot_universe)), fromType = 'UNIPROT', toType = "ENTREZID", OrgDb = base.get('org.Hs.eg.db'))
        universe_prot_entrez_ids =  universe_prot_uniprot_to_entrez_id_mapping[1]

        e_reactome = ReactomePA.enrichPathway(
        gene = test_prot_entrez_ids,
        universe = universe_prot_entrez_ids,
        pAdjustMethod="BH",
        pvalueCutoff  = kwargs.get('pval_cutoff'),
        qvalueCutoff  = kwargs.get('qval_cutoff'),
        readable = True
        )

        utils_package.write_csv(e_reactome, out_file)

        df = pd.read_csv(out_file, index_col=0)

        if not df.empty:
            uniprot_map = kwargs['gene_to_uniprot']
            df['geneName'] = df['geneID']
            df['geneID'] = df['geneName'].apply(lambda x: '/'.join([uniprot_map.get(p,p) for p in x.split('/')]))
        df.to_csv(out_file)

    df = pd.read_csv(out_file, index_col=0)

    return df



def run_clusterProfiler_DAVID(
        prots_to_test, out_dir,annotation_list, prot_universe=None, forced=False, **kwargs):
    """
    parameters: prots_to_test: the list of predicted proteins(UniprotID)
                out_dir: Directory to write the output i.e. enrichment results
                prot_universe: List of background proteins or protein universe(UniprotID)
                annotation_list: For which annotations(e.g. UP_TISSUE, GOTERM_BP_DIRECT) you want to run this analysis
                forced: True means overwrite already existing results, False means use the existing results
                **kwargs: The clusterProfiler result only includes geneID, so for geneNames we are passing the 'uniprot_to_gene map' via kwargs.
                          This uniprotID to geneName mapping has to created outside this function.
    *returns*: a DataFrames of the enrichement of DAVID pathways/onts
    """
    os.makedirs(out_dir, exist_ok=True)
    # now load those results and make a table
    # TODO make this a seting
    print("Running enrich DAVID from clusterProfiler")
    if prot_universe is None:
        print("ERROR: default prot universe not yet implemented. Quitting")
        sys.exit()


    for annotation in annotation_list:
        out_file = "%s/enrich-%s-%s.csv" % (out_dir,annotation,str(kwargs.get('pval_cutoff')).replace('.','_'))
        if forced is False and os.path.isfile(out_file):
            print("\t%s already exists. Use --force-run to overwrite" % (out_file))
            continue
        client = None
        if client is None:
            print("Setting up david client")
            client = david_client.DAVIDClient()
            client.set_category(annotation)

        client.setup_inputs(','.join(prots_to_test), idType='UNIPROT_ACCESSION', listName='gene')
        client.setup_universe(','.join(prot_universe),idType = 'UNIPROT_ACCESSION', listName = 'universe')

        client.build_functional_ann_chart()

        client.write_functional_ann_chart(out_file)

        df = pd.read_csv(out_file, sep='\t')

        if 'GOTERM' in annotation:
            df['ID'] = df['Term'].apply(lambda x: x.split('~')[0])
            df['Description'] = df['Term'].apply(lambda x: x.split('~')[1])
        else:
            df['ID'] = df['Term']
            df['Description'] = df['Term']

        df['BgRatio'] = ['/'.join(x) for x in zip(list(df['Pop Hits'].astype(str)),list(df['Pop Total'].astype(str)))]
        df.drop(['Pop Hits','Pop Total'], axis =1, inplace = True)

        df['GeneRatio'] = ['/'.join(x) for x in zip(list(df['Count'].astype(str)),list(df['List Total'].astype(str)))]
        df.drop('List Total', axis =1, inplace = True)

        df['p.adjust'] = df['Benjamini']
        df.drop('Benjamini', axis =1, inplace = True)

        df['geneID'] = df['Genes'].apply(lambda x: '/'.join(x.split(', ')))
        df['pvalue'] = df['Pvalue']

        gene_map = kwargs['uniprot_to_gene']
        df['geneName'] = df['geneID'].apply(lambda x: '/'.join([gene_map.get(p,p) for p in x.split('/')]))

        df['index'] = df['ID']
        df = df.set_index('index')

        df.drop(['Category','%','Pvalue','Genes','Fold Enrichment','Bonferroni','FDR','Term'], axis =1, inplace = True)

        print(df.columns)

        columns = ['ID', 'Description','GeneRatio', 'BgRatio','pvalue','p.adjust','geneID','Count','geneName']

        df = df[columns]

        df.to_csv(out_file)

    ann_dfs = {ann: pd.DataFrame() for ann in annotation_list}
    for ann in annotation_list:
        out_file = "%s/enrich-%s-%s.csv" % (out_dir,ann,str(kwargs.get('pval_cutoff')).replace('.','_'))
        print("\treading %s" % (out_file))
        df = pd.read_csv(out_file, index_col=0)
        print(df.index)
        ann_dfs[ann] = df
    return ann_dfs

def load_gene_names(id_mapping_file):
    """
    parameters: id_mapping_file: a file containing Uniprot_ID and geneName mapping
    returns: a dictionary where uniprot_id is the key and genename is the value
    """
    df = pd.read_csv(id_mapping_file, sep='\t', header=0)
    ## keep only the first gene for each UniProt ID
    uniprot_to_gene = {p: genes.split(' ')[0] for p, genes in zip(df['Entry'], df['Gene names'].astype(str))}
    if 'Protein names' in df.columns:
        uniprot_to_prot_names = dict(zip(df['Entry'], df['Protein names'].astype(str)))
        #node_desc = {n: {'Protein names': uniprot_to_prot_names[n]} for n in uniprot_to_prot_names}
    return uniprot_to_gene

def load_uniprot(id_mapping_file):
    """
    parameters: id_mapping_file: a file containing Uniprot_ID and geneName mapping
    returns: a dictionary where genename is the key and uniprot_id is the value
    """
    df = pd.read_csv(id_mapping_file, sep='\t', header=0)
    ## keep only the first gene for each UniProt ID
    gene_to_uniprot = {genes.split(' ')[0]: p for p, genes in zip(df['Entry'], df['Gene names'].astype(str))}

    return gene_to_uniprot

if __name__ == "__main__":
    kwargs = parse_args()
    main(**kwargs)
