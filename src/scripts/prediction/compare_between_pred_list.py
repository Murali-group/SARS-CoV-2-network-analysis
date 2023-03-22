import pandas as pd
from scipy.stats import pearsonr

def jaccard(list1, list2):
    return len(set(list1).intersection(set(list2))) / len(set(list1).union(set(list2)))

def pearson_correlations(list1, list2):
    #write code for finding pearson correlations coefficient here
    return pearsonr(list1, list2)

def main(alpha):
    # given an alpha, this function will compute the jaccard index between two predicted list. Hard coded the pred_files.
    network = 'HI-union'
    project_dir = 'Provenance-Tracing'
    maxi = str(1000)

    # network = 'biogrid-y2h'
    # project_dir = 'SARS-CoV-2-Provenance'
    # maxi = str(100)

    # pred_file_pagerank = "/data/tasnina/"+project_dir+"/SARS-CoV-2-network-analysis/outputs/networks/" +network+ \
    #                      "/2020-03-sarscov2-human-ppi-ace2/rwr-PR/pred-scores-alpha" + str(alpha) + \
    #                      "-eps0_0001-maxi"+maxi+"-2020-03-sarscov2-human-ppi-ace2.txt"

    pred_file_pagerankmatrix = "/data/tasnina/"+project_dir+"/SARS-CoV-2-network-analysis/outputs/networks/" +network+ \
                               "/2020-03-sarscov2-human-ppi-ace2/rwr-blessyPR/pred-scores-alpha" + str(alpha) + \
                               "-eps0_0001-maxi"+maxi+"-2020-03-sarscov2-human-ppi.txt"

    # pred_file_jeff = "/data/tasnina/"+project_dir+"/SARS-CoV-2-network-analysis/outputs/networks/" +network+ \
    #                  "/2020-03-sarscov2-human-ppi-ace2/rwr-jeff/pred-scores-a" + str(alpha) + \
    #                  "-2020-03-sarscov2-human-ppi-ace2.txt"

    pred_file_nure = "/data/tasnina/"+project_dir+"/SARS-CoV-2-network-analysis/outputs/networks/" +network+ \
                     "/2020-03-sarscov2-human-ppi-ace2/rwr-nure/pred-scores-alpha" + str(alpha) + \
                     "-eps0_0001-maxi"+maxi+"-2020-03-sarscov2-human-ppi.txt"

    top_k = 5000
    # pred_df_pr = pd.read_csv(pred_file_pagerank, sep='\t')
    pred_df_prm = pd.read_csv(pred_file_pagerankmatrix, sep='\t')
    pred_df_nure = pd.read_csv(pred_file_nure, sep='\t')
    # pred_df_jeff = pd.read_csv(pred_file_jeff, sep='\t')


    print('Jaccard Index: ')
    # print('PR vs PRM: ', jaccard(list(pred_df_pr['prot'])[0:top_k], list(pred_df_prm['prot'])[0:top_k]))
    # print('PR vs Jeffs: ', jaccard(list(pred_df_pr['prot'])[0:top_k], list(pred_df_jeff['prot'])[0:top_k]))
    print('PRM vs Nures: ', jaccard(list(pred_df_prm['prot'])[0:top_k], list(pred_df_nure['prot'])[0:top_k]))
    # print('Nures vs jeffs ', jaccard(list(pred_df_nure['prot'])[0:top_k], list(pred_df_jeff['prot'])[0:top_k]))


    #sort dfs according to the prot names and then compute pearsons correlations on the sorted list
    # pred_df_pr = pred_df_pr.sort_values(by=['prot'])
    pred_df_prm = pred_df_prm[0:top_k].sort_values(by=['prot'])
    # pred_df_jeff = pred_df_jeff.sort_values(by=['prot'])
    pred_df_nure = pred_df_nure[0:top_k].sort_values(by=['prot'])

    print('Pearsons correlations: ')
    # print('PR vs PRM: ', pearson_correlations(list(pred_df_pr['score'])[0:top_k], list(pred_df_prm['score'])[0:top_k]))
    # print('PR vs Jeffs: ', pearson_correlations(list(pred_df_pr['score'])[0:top_k], list(pred_df_jeff['score'])[0:top_k]))
    print('PRM vs Nures: ', pearson_correlations(list(pred_df_prm['score'])[0:top_k], list(pred_df_nure['score'])[0:top_k]))
    # print('Nures vs jeffs ', pearson_correlations(list(pred_df_nure['score'])[0:top_k], list(pred_df_jeff['score'])[0:top_k]))



for alpha in ['0_01', '0_1', '0_25', '0_5', '0_75', '0_9', '0_99']:
    print('\nalpha: ', alpha)
    main(alpha)

