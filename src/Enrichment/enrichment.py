"""
Script to test for enrichment of any given list of genes (UniProt IDs)
"""

import argparse
import yaml
from collections import defaultdict
import os
import sys
#from tqdm import tqdm
import copy
import time
#import numpy as np
#from scipy import sparse
import pandas as pd
#import subprocess

print("importing R packages")
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
utils_package = importr("utils")
clusterProfiler = importr('clusterProfiler')
ReactomePA = importr('ReactomePA')
# this package is needed by clusterProfiler, but importing it directly gives an error, so just import it into R
#print("importing orgDB")
#orgDB = importr('org.Hs.eg.db')
base = importr('base')
base.require('org.Hs.eg.db')

# packages in this repo
sys.path.insert(1, '/home/tasnina/SARS-CoV-2-network-analysis')
from src.utils import parse_utils as utils
from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        #config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    # TODO check to make sure the inputs are correct in config_map

    return config_map, kwargs


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
    #group.add_argument('--out-dir', type=str,
    #                   help="path/to/output directory for enrichemnt files")
    #group.add_argument('--compare-krogan-nodes',
    #                   help="Also test for enrichment of terms when using the Krogan nodes.")
    # Should be specified in the config file
    #group.add_argument('--gmt-file', append=True,
    #                   help="Test for enrichment using the genesets present in a GMT file.")
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


def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    # load the namespace mappings
    uniprot_to_gene = None
    gene_to_uniprot = None
    if kwargs.get('id_mapping_file'):
        # if kwargs.get('id_mapping_file'):
        uniprot_to_gene = load_gene_names(kwargs.get('id_mapping_file'))
        kwargs['uniprot_to_gene'] = uniprot_to_gene

        gene_to_uniprot = load_uniprot(kwargs.get('id_mapping_file'))
        kwargs['gene_to_uniprot'] = gene_to_uniprot

    genesets_to_test = config_map.get('genesets_to_test')
    if genesets_to_test is None or len(genesets_to_test) == 0:
        print("ERROR: no genesets specified to test for overlap. " +
              "Please add them under 'genesets_to_test'. \nQuitting")
        sys.exit()

    # first load the gene sets
    geneset_groups = {}
    for geneset_to_test in genesets_to_test:
        name = geneset_to_test['name']
        gmt_file = "%s/genesets/%s/%s" % (
            input_dir, name, geneset_to_test['gmt_file'])
        if not os.path.isfile(gmt_file):
            print("WARNING: %s not found. skipping" % (gmt_file))
            sys.exit()

        geneset_groups[name] = utils.parse_gmt_file(gmt_file)

    df = pd.read_csv(kwargs['prot_list_file'], sep='\t', header=None)
    prots_to_test = list(df[df.columns[0]])
    print("%d prots for which to test enrichment. (top 10: %s)" % (len(prots_to_test), prots_to_test[:10]))
    prot_universe = None
    # load the protein universe file
    if kwargs.get('prot_universe_file') is not None:
        df = pd.read_csv(kwargs['prot_universe_file'], sep='\t', header=None)
        prot_universe = df[df.columns[0]]
        print("\t%d prots in universe" % (len(prot_universe)))
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
    # TODO figure out which genesets to test
    #for ont, df in [('BP', bp_df), ('MF', mf_df), ('CC', cc_df)]:
    #    all_dfs[ont] = pd.concat([all_dfs[ont], df])


def run_clusterProfiler_GO(
        prots_to_test, out_dir, prot_universe=None, forced=False, **kwargs):
    """

    *returns*: a list of DataFrames of the enrichement of BP, MF, and CC
    """
    os.makedirs(out_dir, exist_ok=True)
    #out_file = "%s/enrichGO_BP.csv" % (out_dir)
    #if not kwargs.get('force_run') and os.path.isfile(out_file):
    #    print("%s already exists. Use --force-run to overwrite" % (out_file))
    #else:
    #    clusterProfiler_command = "Rscript src/Enrichment/prediction_GO_enrichment.R %s %s" % (
    #        pred_filtered_file, out_dir)
    #    utils.run_command(clusterProfiler_command)
    #    print("enriched GO terms files: %s" % (out_dir))
    #
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
        # converting doesn't seem to be working, so just write to file then read to file
        #ego_BP = ro.conversion.rpy2py(ego_BP)
        #with localconverter(ro.default_converter + pandas2ri.converter):
        #  df = ro.conversion.rpy2py(ego_BP)
        # print("\twriting %s" % (out_file))
        utils_package.write_table(ego,out_file, sep=",")

    ont_dfs = []
    for ont in ['BP', 'MF', 'CC']:
        out_file = "%s/enrich-%s-%s.csv" % (out_dir,ont,str(kwargs.get('pval_cutoff')).replace('.','_'))
        print("\treading %s" % (out_file))
        df = pd.read_csv(out_file, index_col=0)
        # add the gene names if specified
        # if kwargs.get('uniprot_to_gene'):
        gene_map = kwargs['uniprot_to_gene']
        df['geneName'] = df['geneID'].apply(lambda x: '/'.join([gene_map.get(p,p) for p in x.split('/')]))
        df.to_csv(out_file)
        #df.columns = ["%s-k%s-%s-%s" % (alg, 200, dataset_name, col) for col in df.columns]
        # print(df.head())
        ont_dfs.append(df)
    return ont_dfs

def run_clusterProfiler_KEGG(
        prots_to_test, out_dir, prot_universe=None, forced=False, **kwargs):
    """
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
        gene_map = kwargs['uniprot_to_gene']
        df['geneName'] = df['geneID'].apply(lambda x: '/'.join([gene_map.get(p,p) for p in x.split('/')]))
        df.to_csv(out_file)

    df = pd.read_csv(out_file, index_col=0)

    return df

def run_ReactomePA_Reactome(
        prots_to_test, out_dir, prot_universe=None, forced=False,**kwargs):
    """
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
        uniprot_map = kwargs['gene_to_uniprot']
        df['geneName'] = df['geneID']
        df['geneID'] = df['geneName'].apply(lambda x: '/'.join([uniprot_map.get(p,p) for p in x.split('/')]))
        df.to_csv(out_file)

    df = pd.read_csv(out_file, index_col=0)

    return df

def get_k_to_test(dataset, **kwargs):
    k_to_test = dataset['k_to_test'] if 'k_to_test' in dataset else kwargs.get('k_to_test', [])
    range_k_to_test = dataset['range_k_to_test'] if 'range_k_to_test' in dataset \
                        else kwargs.get('range_k_to_test')
    if range_k_to_test is not None:
        k_to_test += list(range(
            range_k_to_test[0], range_k_to_test[1], range_k_to_test[2]))
    # if nothing was set, use the default value
    if k_to_test is None or len(k_to_test) == 0:
        k_to_test = [100]
    return k_to_test


def load_gene_names(id_mapping_file):
    df = pd.read_csv(id_mapping_file, sep='\t', header=0)
    ## keep only the first gene for each UniProt ID
    uniprot_to_gene = {p: genes.split(' ')[0] for p, genes in zip(df['Entry'], df['Gene names'].astype(str))}
    if 'Protein names' in df.columns:
        uniprot_to_prot_names = dict(zip(df['Entry'], df['Protein names'].astype(str)))
        #node_desc = {n: {'Protein names': uniprot_to_prot_names[n]} for n in uniprot_to_prot_names}
    return uniprot_to_gene

def load_uniprot(id_mapping_file):
    df = pd.read_csv(id_mapping_file, sep='\t', header=0)
    ## keep only the first gene for each UniProt ID
    gene_to_uniprot = {genes.split(' ')[0]: p for p, genes in zip(df['Entry'], df['Gene names'].astype(str))}

    return gene_to_uniprot

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
