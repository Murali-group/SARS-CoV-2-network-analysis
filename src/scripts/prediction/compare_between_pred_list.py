import pandas as pd


def jaccard(list1, list2):
    return len(set(list1).intersection(set(list2))) / len(set(list1).union(set(list2)))

def pearson_correlations(list1, list2):
    #write code for finding pearson correlations coefficient here

def main(alpha):
    # given an alpha, this function will compute the jaccard index between two predicted list. Hard coded the pred_files.

    pred_file_pagerank = "/data/tasnina/SARS-CoV-2-Provenance/SARS-CoV-2-network-analysis/outputs/networks/biogrid-y2h/" + \
                         "2020-03-sarscov2-human-ppi-ace2/rwr-PR/pred-scores-alpha" + str(alpha) + \
                         "-eps0_0001-maxi100-2020-03-sarscov2-human-ppi-ace2.txt"

    pred_file_pagerankmatrix = "/data/tasnina/SARS-CoV-2-Provenance/SARS-CoV-2-network-analysis/outputs/networks/biogrid-y2h/" + \
                               "2020-03-sarscov2-human-ppi-ace2/rwr-blessyPR/pred-scores-alpha" + str(alpha) + \
                               "-eps0_0001-maxi100-2020-03-sarscov2-human-ppi-ace2.txt"

    pred_file_jeff = "/data/tasnina/SARS-CoV-2-Provenance/SARS-CoV-2-network-analysis/outputs/networks/biogrid-y2h/" + \
                     "2020-03-sarscov2-human-ppi-ace2/rwr-jeff/pred-scores-a" + str(alpha) + \
                     "-2020-03-sarscov2-human-ppi-ace2.txt"

    pred_file_nure = "/data/tasnina/SARS-CoV-2-Provenance/SARS-CoV-2-network-analysis/outputs/networks/biogrid-y2h/" + \
                     "2020-03-sarscov2-human-ppi-ace2/rwr-nure/pred-scores-alpha" + str(alpha) + \
                     "-eps0_0001-maxi100-2020-03-sarscov2-human-ppi-ace2.txt"

    top_k = 1000
    pred_df_1 = pd.read_csv(pred_file_pagerank, sep='\t')
    pred_df_2 = pd.read_csv(pred_file_pagerankmatrix, sep='\t')
    print('PR vs PRM: ', jaccard(list(pred_df_1['prot'])[0:top_k], list(pred_df_2['prot'])[0:top_k]))
    print('Pearsons_correlations: PR vs PRM: ', pearson_correlations(list(pred_df_1['prot'])[0:top_k], list(pred_df_2['prot'])[0:top_k]))

    pred_df_1 = pd.read_csv(pred_file_pagerank, sep='\t')
    pred_df_2 = pd.read_csv(pred_file_jeff, sep='\t')
    print('PR vs Jeffs: ', jaccard(list(pred_df_1['prot'])[0:top_k], list(pred_df_2['prot'])[0:top_k]))
    print('Pearsons_correlations: PR vs Jeffs: ', pearson_correlations(list(pred_df_1['prot'])[0:top_k], list(pred_df_2['prot'])[0:top_k]))


    pred_df_1 = pd.read_csv(pred_file_pagerank, sep='\t')
    pred_df_2 = pd.read_csv(pred_file_nure, sep='\t')
    print('PR vs Nures: ', jaccard(list(pred_df_1['prot'])[0:top_k], list(pred_df_2['prot'])[0:top_k]))
    print('Pearsons_correlations: PR vs Nures: ', pearson_correlations(list(pred_df_1['prot'])[0:top_k], list(pred_df_2['prot'])[0:top_k]))


    pred_df_1 = pd.read_csv(pred_file_nure, sep='\t')
    pred_df_2 = pd.read_csv(pred_file_jeff, sep='\t')
    print('Nures vs jeffs ', jaccard(list(pred_df_1['prot'])[0:top_k], list(pred_df_2['prot'])[0:top_k]))
    print('Pearsons_correlations: Nures vs jeffs ', pearson_correlations(list(pred_df_1['prot'])[0:top_k], list(pred_df_2['prot'])[0:top_k]))



for alpha in ['0_01', '0_1', '0_25', '0_5', '0_75', '0_9', '0_99']:
    print('\n\nalpha: ', alpha)
    main(alpha)

