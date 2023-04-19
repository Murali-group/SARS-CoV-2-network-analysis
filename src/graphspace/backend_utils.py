#All the backend codes like: 1. collectung information about nodes and edges will be here.
import os, sys
import pandas as pd
import networkx as nx
from collections import defaultdict
import numpy as np
from src.FastSinkSource.src.utils import go_prep_utils

go_ont = {'P':'BP', 'F':'MF','C':'CC'}
def get_node_score_rank(pred_file, orig_pos):
    '''Input: pred_file contains predicted scores that are sorted descendingly. Source nodes are given rank='N/A'
     columns: ['#term' , 'prot','score']'''
    if not os.path.isfile(pred_file):
        print("Warning: %s not found. skipping" % (pred_file))
        exit
    df = pd.read_csv(pred_file, sep='\t')

    #prepare source prots
    df_pos= df[df['prot'].isin(orig_pos)]
    df_pos['rank'] = ['N/A']*len(df_pos)
    df_pos = df_pos[['prot', 'score', 'rank']]

    #prepare non-source prots
    df_nonpos = df[~df['prot'].isin(orig_pos)]
    df_nonpos.reset_index(inplace=True, drop=True)
    #rank starts from 1
    df_nonpos['rank'] = df_nonpos.index + 1
    df_nonpos = df_nonpos[['prot', 'score', 'rank']]

    #combine source and non-source prots
    df = pd.concat([df_nonpos, df_pos], axis=0)
    return df



def get_node_function(gaf_file):
    '''input: gaf_file contains go terms annotated to proteins
    output:
        1. node_functions_dfs: this will output a dict where keys are GO categories: ['C','F','P']. The value is another
        dict containing prots as keys and set of GO terms annotated to it as values.
        2. Also save the node_functions_dfs to a file
    '''
    prot_goids_by_c, _, _, _ = go_prep_utils.parse_gaf_file(gaf_file)
    node_functions_dfs = pd.DataFrame()
    for c in prot_goids_by_c:
        ont=go_ont[c]
        prots = list(prot_goids_by_c[c].keys())
        go_ids = list(prot_goids_by_c[c].values())
        df = pd.DataFrame({'prot': prots, 'GO_'+ont+'_ID': go_ids})

        df.set_index('prot', inplace=True)
        node_functions_dfs = pd.concat([node_functions_dfs, df], axis=1)

        #check if there is a prot for which we do not have any GO annotation
        for i in range(len(go_ids)):
            if len(go_ids[i])<0:
                print ('no annotation ', ont ,'  ', prots[i])
        print('Done GO category: ', ont)

    node_functions_dfs.reset_index(inplace=True)
    return node_functions_dfs

def get_node_function_most_spec_and_filtered(gaf_file, obo_file, pred_enrich_file, path_enrich_file, force_run=True):
    '''input: gaf_file contains go terms annotated to proteins
    output:
        1. node_functions_dfs: this will output a dict where keys are GO categories: ['C','F','P']. The value is another
        dict containing prots as keys and set of GO terms annotated to it as values.
        2. Also save the node_functions_dfs to a file
    '''

    out_file = os.path.dirname(gaf_file) + '/prot_2_goid_term.tsv'

    if (not os.path.exists(out_file)) or (force_run):
        prot_goids_by_c, _, _, _ = go_prep_utils.parse_gaf_file(gaf_file)


        # go_categories = set(go_id_2_name_df['category'].unique())
        # go_id_2_name_dict = {c:{} for c in go_categories }
        # for category in go_categories:
        #     go_id_2_name_df_c = go_id_2_name_df[go_id_2_name_df['category']==category]
        #     go_id_2_name_dict[category] = dict(zip(go_id_2_name_df_c['GO_ID'], go_id_2_name_df_c['GO_term']))

        #get the most specific terms from a list of GO terms
        go_dags = go_prep_utils.parse_obo_file_and_build_dags(obo_file)

        node_functions_dfs = pd.DataFrame()
        for c in prot_goids_by_c:

            prots = list(prot_goids_by_c[c].keys())
            go_ids = list(prot_goids_by_c[c].values())

            ont=go_ont[c]

            df = pd.DataFrame({'prot': prots, 'GO_'+ont+'_ID': go_ids})

            # Now find the enriched terms in top_preds and top_paths
            ont_pred_enrich_file = pred_enrich_file.replace('ONT', ont)
            ont_pred_enrich_goids = set(pd.read_csv(ont_pred_enrich_file, index_col=None)['ID'])
            df['GO_' + ont + '_ID_pred_enriched'] = df['GO_' + ont + '_ID'].apply(
                lambda x: filter_goids_according_to_enriched_goids(x, ont_pred_enrich_goids))

            ont_path_enrich_file = path_enrich_file.replace('ONT', ont)
            ont_path_enrich_goids = set(pd.read_csv(ont_path_enrich_file, index_col=None)['ID'])
            df['GO_' + ont + '_ID_path_enriched'] = df['GO_' + ont + '_ID'].apply(
                lambda x: filter_goids_according_to_enriched_goids(x, ont_path_enrich_goids))

            # keep only the most specific go_ids for each protein
            df['GO_' + ont + '_ID'] = df['GO_' + ont + '_ID'].apply(
                lambda x: list(go_prep_utils.get_most_specific_terms(x, go_dags[c])))
            df['GO_' + ont + '_ID_pred_enriched'] = df['GO_' + ont + '_ID_pred_enriched'].apply(
                lambda x: list(go_prep_utils.get_most_specific_terms(x, go_dags[c])))
            df['GO_' + ont + '_ID_path_enriched'] = df['GO_' + ont + '_ID_path_enriched'].apply(
                lambda x: list(go_prep_utils.get_most_specific_terms(x, go_dags[c])))

            # #convert go ids to go term names
            # df['GO_'+ont+'_term'] = df['GO_'+ont+'_ID'].apply(lambda x: map_set_elements(x, go_id_2_name_dict))
            # df['GO_'+ont+'_term_pred_enriched'] = df['GO_'+ont+'_ID_pred_enriched'].apply(lambda x: map_set_elements(x, go_id_2_name_dict))
            # df['GO_'+ont+'_term_path_enriched'] = df['GO_'+ont+'_ID_path_enriched'].apply(lambda x: map_set_elements(x, go_id_2_name_dict))

            df.set_index('prot', inplace=True)
            node_functions_dfs = pd.concat([node_functions_dfs, df], axis=1)

            print('Done GO category: ', ont)

        node_functions_dfs.reset_index(inplace=True)
        node_functions_dfs.to_csv(out_file, sep='\t')
    # else:
    #     node_functions_dfs = pd.read_csv(out_file, sep='\t', converters={'GO_BP_ID': pd.eval})

    node_functions_dfs = pd.read_csv(out_file, sep='\t', converters={'GO_BP_ID': pd.eval})

    return node_functions_dfs

def filter_goids_according_to_enriched_goids(initial_go_ids, enrich_goids):

    filt_goids = set(initial_go_ids).intersection(enrich_goids)

    return filt_goids


def handle_get_node_info(pred_file, orig_pos,gaf_file ):
    ''' Output: Dict of dict with outer key 'prot' (uniprot_id). inner keys:
    ['score', 'rank', 'GO_CC_ID', 'GO_MF_ID', 'GO_BP_ID']. '''
    score_rank_df = get_node_score_rank(pred_file, orig_pos)
    score_rank_df.set_index('prot', inplace=True)

    go_df = get_node_function(gaf_file)
    go_df.set_index('prot', inplace=True)

    node_info_df = pd.concat([score_rank_df, go_df], axis=1)
    node_info_dict = node_info_df.to_dict(orient='index')

    return node_info_dict

def map_set_elements(x, mapping_dict):
    '''
    map each element of a set x to some other element when the mapping is given in mapping_dict
    '''

    y = set([mapping_dict[i] for i in x])
    return y

def get_go_id_2_name_mapping(obo_file):
    goid_names_file = obo_file.replace(".obo","-names.txt")  # contains three tab separated cols containing goid, name, category
    go_id_2_name_df = pd.read_csv(goid_names_file, names=['GO_ID', 'GO_term', 'category'], sep='\t')
    go_id_2_name_dict = dict(zip(go_id_2_name_df['GO_ID'], go_id_2_name_df['GO_term']))
    return go_id_2_name_dict